"""Put obstacles beside the parking manoeuvre so recall can be measured.

The park drives recorded before these ran across an open paved square in
Town10HD: the only occupied cells within fifteen metres of the path were
building walls ten to fifteen metres out and three storeys tall. Nothing a
parking manoeuvre would actually hit was in the scene, so a mapper that finds
nothing near the car and one that finds everything scored the same.

The mix is deliberate: parked cars carry their body between 0.3 and 1.4 m,
barriers stand about a metre, cones about 0.7, so the band a ground-projection
camera can see and the band only a taller-object method can see are both
represented.

    python3 spawn_obstacles.py [--layout a|b] [--clear]
"""

import argparse
import math
import time

import carla

# x, y, yaw, blueprint. Offsets are 3 to 4 m from the path -- close enough to
# be inside the cameras' ground band, far enough that the manoeuvre still runs.
#
# The path runs down x = -48.82. Measured against it, the cars sat 2.3 m out
# and the near cones 1.2 m, which is not what the line above says and not
# something the drive survives: a Prius is 1.0 m to the side of its centre and
# a parked car 0.9 m to the side of its own, so 2.3 m leaves 0.4 m of gap and
# the ego caught the first car 5 m in. It went unnoticed because the bridge
# reloaded the world at startup and deleted every obstacle before recording
# began -- no drive here has ever had to get past them. Cars now sit 3.5 m
# out and the near cones 2.5 m, which is the band the comment describes.
#
# The three barriers are the exception and stay where they are: they are the
# wall 18.8 m ahead that the approach noses up to, and 16-8 measures the
# front camera closing on them from 8.5 m to 5.4.
OBSTACLES = [
    # A wall the vehicle noses up to, which is the manoeuvre the front camera
    # exists for.
    (-51.0, -14.0, 90.0, 'static.prop.streetbarrier'),
    (-48.8, -14.0, 90.0, 'static.prop.streetbarrier'),
    (-46.6, -14.0, 90.0, 'static.prop.streetbarrier'),
    # Adjacent bays, either side of the path.
    (-45.3, -3.5, 90.0, 'vehicle.audi.a2'),
    (-45.3, -7.5, 90.0, 'vehicle.nissan.micra'),
    (-52.3, -5.0, -90.0, 'vehicle.seat.leon'),
    (-52.3, -9.5, -90.0, 'vehicle.tesla.model3'),
    # Low objects in the same band, to see whether height as well as presence
    # survives: a cone is 0.7 m where a car body is 1.4.
    (-46.3, -11.5, 0.0, 'static.prop.trafficcone01'),
    (-51.3, -11.5, 0.0, 'static.prop.trafficcone01'),
    # Beside the reversing curve.
    (-44.6, -6.0, 40.0, 'static.prop.streetbarrier'),
    (-43.5, -1.5, 0.0, 'static.prop.trafficcone01'),
]

# A second arrangement over the same stretch of square. Every number tuned
# against the first layout has had one chance to be a property of that layout
# rather than of the method, and nothing measured could tell the two apart.
# Same treatment as layout a: the cars and the cones that stood inside the
# path have been moved out to the band the comment above describes, keeping
# each one's distance along the drive and which side it is on. The two
# barriers that form this layout's wall keep their places.
LAYOUT_B = [
    (-50.4, -13.2, 60.0, 'static.prop.streetbarrier'),
    (-47.4, -14.6, 120.0, 'static.prop.streetbarrier'),
    (-45.8, -10.8, 20.0, 'static.prop.streetbarrier'),
    (-52.2, -2.6, -70.0, 'vehicle.nissan.micra'),
    (-45.4, -5.2, 100.0, 'vehicle.tesla.model3'),
    (-52.2, -7.8, -110.0, 'vehicle.audi.a2'),
    (-45.4, -9.9, 80.0, 'vehicle.seat.leon'),
    (-51.3, -10.4, 0.0, 'static.prop.trafficcone01'),
    (-46.3, -6.6, 0.0, 'static.prop.trafficcone01'),
    (-51.3, -4.6, 0.0, 'static.prop.trafficcone01'),
    (-44.2, -3.2, 25.0, 'static.prop.streetbarrier'),
]

TAG = 'hero_obstacle'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    # 2 cm, and the reason is not clearance -- it is that CARLA refuses a spawn
    # whose collision volume intersects the road, and at 0.00 the seat.leon and
    # tesla.model3 are refused every time (9 of 11 placed). 0.02 places all
    # eleven. It was 0.3 until 08-07, which left every obstacle in every
    # recording hanging in the air with the road visible underneath.
    parser.add_argument('--z', type=float, default=0.02)
    parser.add_argument('--layout', default='a', choices=('a', 'b'))
    parser.add_argument('--no-settle', action='store_true',
                        help='leave the vehicles hanging at --z, the way every '
                             'bag before 08-07 was recorded. Only for '
                             'reproducing that state to measure against')
    parser.add_argument('--clear', action='store_true',
                        help='remove what a previous run put down and stop')
    args = parser.parse_args()
    obstacles = OBSTACLES if args.layout == 'a' else LAYOUT_B

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    library = world.get_blueprint_library()

    # Static props carry no role_name, so tagging cannot find them again and
    # three runs' worth quietly accumulated in the map. Anything of a type this
    # script places, standing inside the area it works in, is ours.
    kinds = {name for _, _, _, name in OBSTACLES + LAYOUT_B}

    def ours(actor):
        if actor.attributes.get('role_name') == TAG:
            return True
        if not any(actor.type_id.startswith(kind.rsplit('.', 1)[0])
                   for kind in kinds):
            return False
        loc = actor.get_location()
        return -60.0 < loc.x < -30.0 and -25.0 < loc.y < 25.0

    # ours() misses props that a previous run left outside the tagged set, and
    # what is left behind is not harmless: an obstacle spawned onto an older
    # one sits on its roof, and ground_projection then reports that roof as the
    # ground. Every reading of "the square is not flat" traced back to this.
    # Sweep the working area for anything of these kinds, tagged or not.
    existing = [actor for actor in world.get_actors() if ours(actor)]
    for actor in existing:
        try:
            actor.destroy()
        except RuntimeError:
            pass
    if existing:
        print(f'removed {len(existing)} obstacles from a previous run')
    if args.clear:
        return

    placed = 0
    spawned = []
    for x, y, yaw, name in obstacles:
        blueprints = library.filter(name)
        if not blueprints:
            print(f'  no blueprint {name}')
            continue
        blueprint = blueprints[0]
        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', TAG)
        # The coordinates above are read off the ground truth topic, which the
        # bridge publishes in ROS convention -- y to the left. CARLA is left
        # handed with y to the right, so placing them here without the flip
        # mirrors every obstacle to the far side of the path. That is what put
        # three scenes' worth of obstacles behind the vehicle, where only the
        # rear camera ever saw them.
        transform = carla.Transform(
            carla.Location(x=x, y=-y, z=args.z), carla.Rotation(yaw=-yaw))
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            print(f'  refused at ({x:.1f}, {y:.1f}): {name}')
            continue
        # Immediately, not after the loop. Physics is on when an actor
        # spawns, and eleven spawns take long enough that a car placed early
        # slides while the rest are still going down: the micra came to rest
        # 1.9 m from where it was asked for, on top of another obstacle at
        # z = 0.343.
        try:
            actor.set_simulate_physics(False)
        except RuntimeError:
            pass
        spawned.append(actor)
        placed += 1
        print(f'  {name} at ({x:.1f}, {y:.1f})')

    # Nothing is lowered here any more, because nothing needs to be: --z is 0.
    #
    # It was 0.3 for years, so an actor could not spawn intersecting the road,
    # and the vehicles then had physics switched off immediately -- which left
    # every obstacle in the scene hanging 0.3 m in the air. The front camera
    # shows it plainly: road visible under the wheels, shadows detached from
    # what casts them. A camera that can see under a parked car carves the
    # ground beneath it free, so the cell the car stands in was scored against
    # a map that had been shown an empty floor.
    #
    # Two fixes were tried and thrown away. Dropping under physics scatters
    # things -- bodies land on slopes and slide, land on each other and bounce;
    # 3 s of settling moved one prop 13.9 m and 0.6 s moved another 43.5 m.
    # Asking ground_projection where the ground is returns the actor's own roof
    # when the ray starts above the actor, and correcting by the bounding box
    # raised everything by a metre.
    #
    # The measurement that settled it is the drop itself: from 0.3 m the
    # vehicles fell 0.298 to 0.319 m, so the road is at z = 0 and these
    # coordinates need no height correction at all. Spawning there works --
    # CARLA only refuses a spawn that intersects another actor, not one resting
    # on the road.
    for actor in spawned:
        try:
            actor.set_simulate_physics(False)
        except RuntimeError:
            pass
    if args.z > 0.05:
        print(f'  warning: --z {args.z} leaves obstacles {args.z} m off the '
              f'ground -- cameras will see underneath them')

    print(f'{placed} of {len(obstacles)} obstacles placed (layout {args.layout})')


if __name__ == '__main__':
    main()
