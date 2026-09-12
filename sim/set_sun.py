"""Move the sun without touching anything else in the world.

Used as a control, not as a countermeasure: the ego's own shadow lies across
the ground band one camera reads, and the question was whether that is what
the phantom cells are made of. Controlling the sun does not fix the problem on
a real vehicle and was never meant to -- and the one experiment run this way
was confounded anyway, because at azimuth 120 the whole square fell into
building shade and the comparison became "ego shadow against half the light".
"""

import argparse

import carla


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--azimuth', type=float)
    parser.add_argument('--altitude', type=float)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    weather = world.get_weather()
    if args.azimuth is not None:
        weather.sun_azimuth_angle = args.azimuth
    if args.altitude is not None:
        weather.sun_altitude_angle = args.altitude
    world.set_weather(weather)
    print(f'sun azimuth {weather.sun_azimuth_angle:.1f} '
          f'altitude {weather.sun_altitude_angle:.1f}')


if __name__ == '__main__':
    main()
