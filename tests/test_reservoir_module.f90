program test_reservoir_module
    use reservoir_regulation_module
    implicit none

    integer, parameter :: dp = kind(1.0d0)
    integer, parameter :: n_sh = 6, n_hq = 6, n_t = 12

    type(ReservoirCurve) :: curve
    real(dp) :: qin(n_t), step_index(n_t)
    real(dp) :: s_prev, s_new, h_new, qout, dt
    integer :: i

    curve%n_sh = n_sh
    curve%n_hq = n_hq
    allocate(curve%storage(n_sh), curve%level_sh(n_sh))
    allocate(curve%level_hq(n_hq), curve%release(n_hq))

    call read_two_columns("test_data/reservoir/storage_level_curve.csv", &
                          curve%storage, curve%level_sh, n_sh)
    call read_two_columns("test_data/reservoir/level_release_curve.csv", &
                          curve%level_hq, curve%release, n_hq)
    call read_two_columns("test_data/reservoir/inflow_test.csv", &
                          step_index, qin, n_t)

    dt = 3600.0_dp
    s_prev = 2.0e6_dp

    write(*,'(A)') "step,inflow_m3s,storage_m3,water_level_m,outflow_m3s"

    do i = 1, n_t
        call reservoir_step( &
            qin=qin(i), dt=dt, s_prev=s_prev, curve=curve, &
            s_new=s_new, h_new=h_new, qout=qout)
        write(*,'(I0,A,F10.3,A,F14.3,A,F10.3,A,F10.3)') &
            i, ",", qin(i), ",", s_new, ",", h_new, ",", qout
        s_prev = s_new
    end do

contains

    subroutine read_two_columns(filename, x, y, n)
        character(len=*), intent(in) :: filename
        integer, intent(in) :: n
        real(dp), intent(out) :: x(n), y(n)
        integer :: unit_id, j
        character(len=512) :: header

        open(newunit=unit_id, file=filename, status="old", action="read")
        read(unit_id,'(A)') header
        do j = 1, n
            read(unit_id,*) x(j), y(j)
        end do
        close(unit_id)
    end subroutine read_two_columns

end program test_reservoir_module
