"""
Simulate a stellar wind around a binary/triple
"""

import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.community.phantom import Phantom
from amuse.community.fi import Fi
from amuse.community.ph4 import Ph4

# from amuse.community.hermite import Hermite
from amuse.couple.bridge import Bridge  # , CalculateFieldForCodes
from amuse.units import units, nbody_system, constants
from amuse.datamodel import Particles
from amuse.io import write_set_to_file

# from amuse.ic.gasplummer import new_plummer_gas_model

from amuse.ic.molecular_cloud import new_molecular_cloud
from amuse.support.console import set_printing_strategy
from amuse.plot.mapper import MapHydro
from amuse.plot.hydro import plot_hydro_and_stars

import config_parsers


np.random.seed(1504)

RUN = "mats_lores_3"
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GiB
# MAX_FILE_SIZE = 1024 * 1024 * 512  # 512 MiB

logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    datefmt="%Y%m%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


# Rewrite from amuse.couple.bridge
class AbstractCalculateFieldForCodes:
    """
    Calculated gravity and potential fields using the particles
    of other codes with the code provided.
    """

    def __init__(self, input_codes, verbose=False, required_attributes=None):
        """
        'verbose' indicates whether to output some run info

        'required_attributes' specifies which particle attributes need to be
        transferred from the input_codes to the code that will calculate the
        field. For example, some codes don't need the velocity. Other codes
        may (wrongly) interpret the radius of the input code as gravitational
        softening. In the latter case
            required_attributes=['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz']
        should prevent the radius of the input codes from being used.
        """
        self.codes_to_calculate_field_for = input_codes
        self.verbose = verbose
        if required_attributes is None:
            self.required_attributes = lambda p, attribute_name: True
        else:
            self.required_attributes = (
                lambda p, attribute_name: attribute_name in required_attributes
            )

    def evolve_model(self, tend, timestep=None):
        """ """

    def get_potential_at_point(self, radius, x, y, z):
        code = self._setup_code()
        try:
            for input_code in self.codes_to_calculate_field_for:
                particles = input_code.gas_particles.copy(
                    filter_attributes=self.required_attributes
                )
                code.particles.add_particles(particles)
            return code.get_potential_at_point(radius, x, y, z)
        finally:
            self._cleanup_code(code)

    def get_gravity_at_point(self, radius, x, y, z):
        code = self._setup_code()
        try:
            for input_code in self.codes_to_calculate_field_for:
                particles = input_code.gas_particles.copy(
                    filter_attributes=self.required_attributes
                )
                code.particles.add_particles(particles)
            return code.get_gravity_at_point(radius, x, y, z)
        finally:
            self._cleanup_code(code)

    def _setup_code(self):
        pass

    def _cleanup_code(self, code):
        pass


class CalculateFieldForCodes(AbstractCalculateFieldForCodes):
    """
    Calculated gravity and potential fields using the particles
    of other codes with the code provided.
    The code is created for every calculation.
    """

    def __init__(self, code_factory_function, input_codes, *args, **kwargs):
        AbstractCalculateFieldForCodes.__init__(self, input_codes, *args, **kwargs)
        self.code_factory_function = code_factory_function

    def _setup_code(self):
        return self.code_factory_function()

    def _cleanup_code(self, code):
        code.stop()


# end rewrite


# set_printing_strategy(
#     "custom",
#     preferred_units=(
#         units.au,
#         units.MSun,
#         units.kms,
#         units.julianyr,
#         units.K,
#         units.LSun,
#     ),
# )


def setup_phantom(phantom_options):
    """
    Set up Phantom for wind simulation
    """
    # sph = Phantom()
    sph = Phantom(redirection="none")

    for item in phantom_options.items():
        if item[0] == "iverbose":
            print(item[0], item[1])
        found = sph.set_phantom_option(item[0], item[1])
        if found:
            print(f"option {item[0]} set to {item[1]}")
        else:
            print(f"could not set option {item[0]}")
    return sph


class PhantomParticles:
    def __init__(self, phantominstance):
        self._instance = phantominstance
        self._gas = Particles()
        self._sinks = Particles()
        self._dm = Particles()
        self._norig = 0
        self._isothermal = True

    @property
    def gas(self):
        return self._gas

    @property
    def sinks(self):
        return self._sinks

    @property
    def dm(self):
        return self._dm

    def update(self):
        """
        Updates the particles with information from Phantom.
        Also adds/deletes particles where needed.
        """
        norig = self._norig
        norig_new = self._instance.get_maximum_particle_index()
        if norig_new > norig:
            number_of_new_particles = norig_new - norig
            new_gas = self.gas.add_particles(Particles(number_of_new_particles))

        active_indices = self._instance.get_active_gas()
        state = self._instance.get_state_sph(
            range(
                norig + 1,
                norig_new + 1,
            )
        )
        new_gas.mass = state[0]
        new_gas.x = state[1]
        new_gas.y = state[2]
        new_gas.z = state[3]
        new_gas.vx = state[4]
        new_gas.vy = state[5]
        new_gas.vz = state[6]
        if self._isothermal:
            new_gas.u = 100 | (units.km / units.s) ** 2
        else:
            new_gas.u = state[7]
        new_gas.h_smooth = state[8]
        new_gas.radius = state[8]
        new_gas.phantomindex = state[9]


CARBON_RICH_PARENT_SPECIES = {
    "H2": 1.0,
    "He": 0.17,
    "CO": 8.0e-4,
    "C2H2": 4.38e-5,
    "HCN": 4.09e-5,
    "N2": 4.0e-5,
    "SiC2": 1.87e-5,
    "CS": 1.06e-5,
    "SiS": 5.98e-6,
    "SiO": 5.02e-6,
    "CH4": 3.5e-6,
    "H2O": 2.55e-6,
    "HCl": 3.25e-7,
    "C2H4": 6.85e-8,
    "NH3": 6.0e-8,
    "HCP": 2.5e-8,
    "HF": 1.7e-8,
    "H2S": 4e-9,
}

OXYGEN_RICH_PARENT_SPECIES = {
    "H2": 1.0,
    "He": 0.17,
    "CO": 3e-4,
    "H2O": 2.15e-4,
    "N2": 4e-5,
    "SiO": 2.71e-5,
    "H2S": 1.75e-5,
    "SO2": 3.72e-6,
    "SO": 3.06e-6,
    "SiS": 9.53e-7,
    "NH3": 6.25e-7,
    "CO2": 3.0e-7,
    "HCN": 2.59e-7,
    "PO": 7.75e-8,
    "CS": 5.57e-8,
    "PN": 1.5e-8,
    "F": 1.0e-8,
    "Cl": 1.0e-8,
}

ATOMIC_WEIGHT = {
    "H": 1 | units.amu,
    "He": 4 | units.amu,
    "C": 12 | units.amu,
    "N": 14 | units.amu,
    "O": 16 | units.amu,
    "F": 19 | units.amu,
    "Si": 28 | units.amu,
    "P": 31 | units.amu,
    "S": 32 | units.amu,
    "Cl": 35.45 | units.amu,
}


def get_molecular_weight_of_species(species_name):
    elements = list(species_name)
    length = len(elements)
    weight = 0 | units.amu
    skip = 0
    for i, char in enumerate(elements):
        if skip > 0:
            skip -= 1
            continue
        unfinished = True
        multiplication = 1
        while unfinished:
            j = 1 + skip
            if i < length - j:
                nextchar = elements[i + j]
                if nextchar.isdigit():
                    multiplication = int(nextchar)
                    skip += 1
                elif nextchar == nextchar.lower():
                    char += nextchar
                    skip += 1
                else:
                    unfinished = False
            else:
                unfinished = False
        this_weight = ATOMIC_WEIGHT[char] * multiplication
        weight += this_weight
        print(f"{multiplication} * {char} added: {this_weight}")
    print(f"total weight: {weight}")
    return weight


def get_mean_molecular_weight(species_number_fractions):
    mmw = 0 | units.amu
    for key in species_number_fractions:
        mmw += species_number_fractions[key] * get_molecular_weight_of_species(key)
    return mmw


def select_new_particles_for_chemistry(
    new_gas,
    star,
    minimum_distance=10**18 | units.cm,
    # outflow_type="oxygen_rich",
    outflow_type="carbon_rich",
):
    select_gas = new_gas[
        (new_gas.position - star.position).lengths() > minimum_distance
    ]

    select_gas.number_density = 1  # FIXME obviously...
    return select_gas


class Chemistry:
    def __init__(self, particles=Particles()):
        from amuse.community.krome import Krome

        self.instance = Krome()
        if isinstance(particles, Particles):
            self._particles = particles
        else:
            print("particles needs to be a Particles instance")
            self._particles = Particles()

    @property
    def particles(self):
        # Needs properties:
        # - number_density
        # - temperature
        # - ionrate
        return self._particles

    def add_particles(self, particles):
        gmmw = (2.33 / 6.02214179e23) | units.g
        if not hasattr(particles, number_density):
            if hasattr(particles, density):
                particles.number_density = particles.density / gmmw
            else:
                print("Needs number_density or density!")

        self.instance.particles.add_particles(particles)
        self.instance.particles.new_channel_to(self._particles).copy()

    def get_species(self):
        return self.instance.species
        # for p in part2:
        #     i = instance.species["E"]
        #     self.assertAlmostEqual(p.abundances[i], 0.000369180975425)
        #     i = instance.species["H+"]
        #     self.assertAlmostEqual(p.abundances[i], 0.0001)
        #     i = instance.species["HE"]
        #     self.assertAlmostEqual(p.abundances[i], 0.0775)
        #     i = instance.species["C+"]
        #     self.assertAlmostEqual(p.abundances[i], 0.000269180975425)
        #     i = instance.species["SI"]
        #     self.assertAlmostEqual(p.abundances[i], 3.2362683404e-05)
        #     i = instance.species["O"]
        #     self.assertAlmostEqual(p.abundances[i], 0.000489828841345)

    def evolve_model(self, time_end):
        self.instance.evolve_model(time_end)
        self.model_time = self.instance.model_time


def makeparts_chem(number_of_particles):
    parts = Particles(number_of_particles)
    parts.number_density = (
        np.random.random(number_of_particles) * 1.0e5 + 1.0e5
    ) | units.cm**-3
    parts.temperature = (np.random.random(number_of_particles) * 500 + 100) | units.K
    parts.ionrate = (
        np.random.random(number_of_particles) * 1.0e-11 + 1.0e-17
    ) | units.s**-1
    return parts


def update_gas_in_code(sph):
    """
    Obtains gas in SPH code (that AMUSE may not yet know about) and returns it
    """
    number_of_particles = sph.get_number_of_particles()
    norig = sph.get_maximum_particle_index()
    print(f"N_SPH: {number_of_particles}, max index: {norig}")
    if number_of_particles == 0:
        return Particles()
    gas_in_code = Particles(number_of_particles)
    state = sph.get_state_sph(
        range(
            1,
            number_of_particles + 1,
        )
    )
    gas_in_code.mass = state[0]
    gas_in_code.x = state[1]
    gas_in_code.y = state[2]
    gas_in_code.z = state[3]
    gas_in_code.vx = state[4]
    gas_in_code.vy = state[5]
    gas_in_code.vz = state[6]
    gas_in_code.u = 100 | (units.km / units.s) ** 2  # state[7]
    gas_in_code.h_smooth = state[8]
    gas_in_code.radius = state[8]
    # particles with h_smooth <= 0 are not active/deleted
    gas_in_code = gas_in_code[gas_in_code.h_smooth > (0 | units.pc)]
    return gas_in_code


def save_files(stars, gas, time, counter):
    """
    Write to save files
    """
    gas_savefile = f"gas-{RUN}-{counter}.amuse"
    stars_savefile = f"stars-{RUN}-{counter}.amuse"
    write_set_to_file(
        gas,
        gas_savefile,
        append_to_file=time > 0 | units.julianyr,
        timestamp=time,
        compression=True,
        overwrite_file=not time > 0 | units.julianyr,
    )
    write_set_to_file(
        stars,
        stars_savefile,
        append_to_file=time > 0 | units.julianyr,
        timestamp=time,
        compression=True,
        overwrite_file=not time > 0 | units.julianyr,
    )
    if os.path.getsize(gas_savefile) > MAX_FILE_SIZE:
        # assuming gas file >> sink file, which is a pretty safe bet
        counter += 1
    return counter


def plot(
    stars,
    gas,
    i,
    fig,
    ax,
    axes="xyz",
    mapper=None,
    reuse_mapper=False,
    vel_center=0 | units.kms,
    vel_range=1 | units.kms,
    width=100 | units.au,
    show=False,
    follow_gas_particles=False,
):
    """
    Nice sph plotting
    """
    gas_for_mapper = gas.empty_copy()
    gas.new_channel_to(gas_for_mapper).copy_attributes(
        ["x", "y", "z", "mass", "h_smooth", "u"]
    )
    # is_active = gas_for_mapper.h_smooth > (0 | units.m)
    # gas_for_mapper = gas_for_mapper[is_active]
    # gas_for_mapper = gas_for_mapper[
    #     gas_for_mapper.vz < (vel_center + vel_range)
    # ]
    # gas_for_mapper = gas_for_mapper[
    #     gas_for_mapper.vz > (vel_center - vel_range)
    # ]
    print(f"Active gas: {len(gas_for_mapper)}")
    print(f"Mass per gas particle: {gas_for_mapper[0].mass.in_(units.MSun)}")
    # print(gas[0])
    # print(gas[1])
    # print(gas[-1])
    gas_for_mapper.radius = gas_for_mapper.h_smooth
    if mapper is None:
        mapper = MapHydro(
            gas=gas_for_mapper,
            stars=stars.copy(),
        )
        mapper.width = width
    # print("\n\n", stars[0], "\n\n")
    mapper.axes = axes
    # for plot in ["column density", "temperature"]:
    for plot_property in [
        "column density",
        "temperature",
    ]:
        gasplot = plot_hydro_and_stars(
            mapper,
            length_unit=units.au,
            fig=fig,
            ax=ax,
            plot=plot_property,
            vmin=2 if plot_property == "column density" else None,
            vmax=4 if plot_property == "column density" else None,
        )
        if follow_gas_particles:
            select_gas = gas[gas.key % 1000 == 0]
            ax.scatter(
                getattr(select_gas, axes[0]).value_in(units.au),
                getattr(select_gas, axes[1]).value_in(units.au),
                s=10,
                marker="x",
                color="red",
                edgecolors="none",
            )
        if plot_property == "column density":
            plot_property = "column_density"
        if not show:
            plt.savefig(f"plot_{RUN}_{plot_property}_{axes[:2]}-{i:06d}.png")
        else:
            plt.show()
        ax.cla()
    if reuse_mapper:
        return mapper
    mapper.stop()
    return None


def heating(gas, stars):
    # placeholder
    return


def find_gas_inside_sphere(
    gas,
    radius,
    center=[0, 0, 0] | units.au,
):
    distance = (gas.position - center).lengths()
    return gas[distance < radius]


def find_gas_outside_sphere(
    gas,
    radius,
    center=[0, 0, 0] | units.au,
):
    distance = (gas.position - center).lengths()
    return gas[distance > radius]


def temperature_to_u(
    temperature,
    gmmw=(2.33 / 6.02214179e23) | units.g,
    # gmmw=gas_mean_molecular_weight(),
):
    """
    Converts temperature to internal energy, assuming a specified mean molecular
    weight, and returns the internal energy
    """
    internal_energy = 3.0 * constants.kB * temperature / (2.0 * gmmw)
    return internal_energy


def run_simulation():
    """
    Run stellar wind simulation
    """
    # bridge_gravity = True
    bridge_gravity = False

    plot_things = True
    save_things = True
    # create initial conditions
    # stars = create_binary()
    stars, phantom_options = config_parsers.read_parse_config(
        wind_in_file="wind_lores.in",
        wind_setup_file="wind_lores.setup",
    )
    print(stars)

    # setup Phantom
    sph = setup_phantom(
        phantom_options,
    )
    print(sph.parameters)
    sph.parameters.gamma = 1.2

    sph.parameters.time_step = 0.5 * sph.parameters.time_step
    time_step = sph.parameters.time_step
    converter = nbody_system.nbody_to_si(0.01 * time_step, 1 | units.au)
    system = Bridge()
    system.timestep = time_step  # 0.05 | units.julianyr

    def new_field_gravity_code(
        code=Fi,
    ):
        "Creates and returns a new field tree code"
        result = code(
            converter,
            redirection="none",
            mode="openmp",
        )
        return result

    def new_field_code(
        code,
    ):
        result = CalculateFieldForCodes(
            new_field_gravity_code,
            [code],
            verbose=False,
        )
        return result

    if bridge_gravity:
        converter = nbody_system.nbody_to_si(0.01 * time_step, 1 | units.au)
        gravity = Ph4(converter)
        gravity.parameters.timestep_parameter = 0.03
        gravity.parameters.force_sync = 1
        stars_in_gravity = gravity.particles.add_particles(stars)
        stars_in_hydro = sph.sink_particles.add_particles(stars)
        stars_in_hydro.temperature = stars.temperature
        stars_in_hydro.luminosity = stars.luminosity
        sph.commit_particles()
        system.add_system(
            gravity,
            partners=[
                new_field_code(sph),
            ],
            do_sync=True,
        )
        system.add_system(
            sph,
            partners=[],
            do_sync=True,
        )
        stars_in_code = stars_in_gravity
        # sph_to_gravity = stars_in_hydro.new_channel_to(stars_in_gravity)
        # gravity_to_sph = stars_in_gravity.new_channel_to(stars_in_hydro)
    else:
        stars_in_code = sph.sink_particles.add_particles(stars)
        stars_in_code.temperature = stars.temperature
        stars_in_code.luminosity = stars.luminosity
        sph.commit_particles()
        print("DONE TO THIS POINT")
        system.add_system(sph, ())

    file_counter = 0
    i = 0
    t = 0 | units.Myr

    number_of_sph_particles = sph.get_number_of_particles()
    # gas_in_code = update_gas_in_code(sph)
    gas_in_code = Particles()
    sph.update_gas_particle_set()
    sph.gas_particles.synchronize_to(gas_in_code)
    print(sph.gas_particles)
    print(gas_in_code)
    # gas_in_code.u = 100 | (units.km / units.s) ** 2
    # gas_in_code = sph.gas_particles
    r_init = (gas_in_code.position - stars[0].position).lengths()
    v_init = (gas_in_code.velocity - stars[0].velocity).lengths()
    # print(f"Initial position: {r_init.mean()} {r_init.std()}")
    # print(f"Initial position minmax: {r_init.min()} {r_init.max()}")
    # print(f"Initial velocity: {v_init.mean()} {v_init.std()}")

    print(f"NSPH: {number_of_sph_particles}, {len(gas_in_code)}")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, aspect=1)
    if plot_things and len(gas_in_code) > 0:
        mapper = plot(
            stars,
            gas_in_code,
            0,
            fig,
            ax,
            axes="xyz",
            reuse_mapper=True,
        )
        mapper.rotate(theta=90 | units.deg)
        plot(
            stars,
            gas_in_code,
            0,
            fig,
            ax,
            axes="xzy",
            mapper=mapper,
        )
    if save_things:
        file_counter = save_files(stars, gas_in_code, t, file_counter)
    system.timestep = sph.parameters.time_step
    initial_mass = stars.mass

    dx = 0 | units.m
    dy = 0 | units.m
    dz = 0 | units.m
    max_offset = 0 | units.julianyr
    # loop_time_step = 0.5 | units.julianyr
    loop_time_step = sph.parameters.time_step
    t = 0 | units.julianyr
    phantom_time_unit = sph.get_unit_time()
    print(phantom_time_unit)
    for i in range(1000):
        t = t + loop_time_step
        # timestep may change?!
        # system.timestep = sph.parameters.time_step
        # system.evolve_model(t)
        print(f"AMUSE: EVOLVING TO TIME {t} ({t / phantom_time_unit})")
        sph.evolve_model(t)
        print(f"AMUSE: EVOLVED TO TIME {t} ({t / phantom_time_unit})")
        # s = sph.get_state_sph(1)
        # print(((s[1]**2 + s[2]**2 + s[3]**2)**0.5).in_(units.au))

        loop_time_step = sph.parameters.time_step
        # print(t, system.model_time, loop_time_step)
        print(t, sph.model_time, loop_time_step)
        # sys.exit()
        if bridge_gravity:
            if abs(sph.model_time - gravity.model_time) > max_offset:
                max_offset = abs(sph.model_time - gravity.model_time)
            print(
                f"Time: requested={t}"
                f" system={system.model_time} gravity={gravity.model_time}"
                f" sph={sph.model_time}"
                f" offset={sph.model_time - gravity.model_time}"
                f" max offset={max_offset} ({max_offset/time_step} time step)"
            )
            if abs(t - sph.model_time) > 0.5 * time_step:
                print("SPH time too far off")
                sys.exit()
            if abs(t - gravity.model_time) > 0.5 * time_step:
                print("Gravity time too far off")
                sys.exit()

            dx += abs(stars_in_hydro.x - stars_in_gravity.x)
            dy += abs(stars_in_hydro.y - stars_in_gravity.y)
            dz += abs(stars_in_hydro.z - stars_in_gravity.z)
            print(
                f"Cumulative offsets in positions (hydro vs gravity): "
                f" {dx} {dy} {dz}"
            )
            g2h = stars_in_gravity.new_channel_to(stars_in_hydro)
            g2h.copy_attributes(["x", "y", "z", "vx", "vy", "vz"])
            h2g = stars_in_hydro.new_channel_to(stars_in_gravity)
            h2g.copy_attributes(
                [
                    "mass",
                ]
            )
        stars_in_code.new_channel_to(stars).copy_attributes(
            ["x", "y", "z", "vx", "vy", "vz"]
        )

        # gas_in_code = update_gas_in_code(sph)
        sph.update_gas_particle_set()
        # gas_in_code = sph.gas_particles
        sph.gas_particles.synchronize_to(gas_in_code)
        sph.gas_particles.new_channel_to(gas_in_code).copy()
        # gas_in_code.u = 100 | (units.km / units.s) ** 2

        print(f"TIME: {system.model_time.in_(units.julianyr)}**********")
        print(f"SPH: {len(gas_in_code)}")
        if not gas_in_code.is_empty():
            print(f"U_mean: {gas_in_code.u.mean()}, sigma: {gas_in_code.u.std()}")
        print(
            f"MDOT: "
            f"star1: {initial_mass[0] - stars[0].mass} "
            f"rate: {(initial_mass[0] - stars[0].mass) / sph.model_time} "
            # f"star2: {minit[1] - stars[1].mass}"
        )
        print(
            f"Gas in code: {gas_in_code.total_mass()}, {gas_in_code.position.lengths().mean()}"
        )
        # plot every 'plot_every' time steps
        plot_every = 1
        j = int(i / plot_every)
        plot_things = bool(i % plot_every == 0)
        if plot_things and len(gas_in_code) > 0:
            mapper = plot(
                stars,
                gas_in_code,
                j,
                fig,
                ax,
                axes="xyz",
                reuse_mapper=True,
            )
            mapper.rotate(theta=90 | units.deg)
            plot(
                stars,
                gas_in_code,
                j,
                fig,
                ax,
                axes="xzy",
                mapper=mapper,
            )
        # save every 'save_every' time steps
        save_every = 1
        save_things = bool(i % save_every == 0)
        if save_things:
            file_counter = save_files(stars_in_code, gas_in_code, t, file_counter)
    return


def main():
    run_simulation()


if __name__ == "__main__":
    main()
