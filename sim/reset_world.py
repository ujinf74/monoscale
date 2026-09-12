"""Return the simulator to a known state between recordings.

Destroys the ego, any sensors still attached, and the props a previous run
left behind. The last of those is not housekeeping: a run that inherited
barriers from an earlier scene drove into them 18.8 m in and four recordings
were made and scored before the three-metre path length gave it away.
"""

import argparse

import carla

AREA = (-60.0, -30.0, -25.0, 25.0)  # x_min, x_max, y_min, y_max


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    removed = 0
    for actor in world.get_actors():
        keep = True
        if actor.type_id.startswith(('vehicle.', 'sensor.')):
            keep = False
        elif actor.type_id.startswith('static.prop.'):
            loc = actor.get_location()
            keep = not (AREA[0] < loc.x < AREA[1] and AREA[2] < loc.y < AREA[3])
        if keep:
            continue
        try:
            actor.destroy()
            removed += 1
        except RuntimeError:
            pass
    print(f'removed {removed} actors; world is asynchronous again')


if __name__ == '__main__':
    main()
