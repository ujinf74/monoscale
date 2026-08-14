from monoscale_occupancy_grid_map.occupancy import LogOddsGrid


def test_ray_marks_free_space_and_occupied_endpoint():
    grid = LogOddsGrid(1.0, 10, 10, 0.0, 0.0, 1.0, 1.0)

    grid.integrate_ray((1.5, 1.5), (4.5, 1.5), occupied=True)
    values = grid.message_values(0.0)

    assert values[1, 1] == 0
    assert values[1, 2] == 0
    assert values[1, 3] == 0
    assert values[1, 4] == 100


def test_occupied_point_does_not_carve_free_space_on_the_way_in():
    grid = LogOddsGrid(1.0, 10, 10, 0.0, 0.0, 1.0, 1.0)

    grid.integrate_ray((1.5, 1.5), (4.5, 1.5), occupied=False)
    grid.integrate_point((2.5, 1.5))
    grid.integrate_point((2.5, 1.5))
    values = grid.message_values(0.0)

    assert values[1, 2] == 100
    # The cells the free ray swept stay free; only the endpoint cell changed.
    assert values[1, 1] == 0
    assert values[1, 3] == 0


def test_point_outside_the_grid_is_ignored():
    grid = LogOddsGrid(1.0, 4, 4, 0.0, 0.0, 1.0, 1.0)

    grid.integrate_point((99.0, 99.0))

    assert (grid.message_values(0.0) == -1).all()


def test_raising_the_threshold_demands_a_second_observation():
    strict = LogOddsGrid(1.0, 4, 4, 0.0, 0.0, 0.45, 0.85, occupied_probability=0.8)
    lenient = LogOddsGrid(1.0, 4, 4, 0.0, 0.0, 0.45, 0.85, occupied_probability=0.65)

    for grid in (strict, lenient):
        grid.integrate_point((1.5, 1.5))

    assert lenient.message_values(0.0)[1, 1] == 100
    assert strict.message_values(0.0)[1, 1] != 100

    strict.integrate_point((1.5, 1.5))
    assert strict.message_values(0.0)[1, 1] == 100
