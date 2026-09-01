module reservoir_regulation_module
    !! 栅格水文模型中的水库调蓄模块。
    !!
    !! 控制关系：
    !!   S(t+1) = S(t) + [Qin(t) - Qout(t+1)] * dt
    !!   H(t+1) = f[S(t+1)]
    !!   Qout(t+1) = g[H(t+1)]

    implicit none
    private

    public :: ReservoirCurve
    public :: reservoir_step

    integer, parameter :: dp = kind(1.0d0)

    type :: ReservoirCurve
        integer :: n_sh = 0
        integer :: n_hq = 0
        real(dp), allocatable :: storage(:)       ! 库容值，例如 m3
        real(dp), allocatable :: level_sh(:)      ! 与库容对应的水位，m
        real(dp), allocatable :: level_hq(:)      ! 水位-下泄流量曲线中的水位，m
        real(dp), allocatable :: release(:)       ! 下泄流量，m3/s
    end type ReservoirCurve

contains

    subroutine reservoir_step(qin, dt, s_prev, curve, s_new, h_new, qout, max_iter, tol)
        !! 将水库状态推进一个模型计算时步。
        !!
        !! 输入参数
        !!   qin      ：当前时步的入库流量，m3/s
        !!   dt       ：模型计算时步，s
        !!   s_prev   ：时步开始时的水库库容，m3
        !!   curve    ：水库 S-H 曲线和 H-Q 曲线
        !!
        !! 输出参数
        !!   s_new    ：时步结束时的水库库容，m3
        !!   h_new    ：时步结束时的水库水位，m
        !!   qout     ：调蓄后的下泄流量，并汇入下游栅格，m3/s

        real(dp), intent(in) :: qin, dt, s_prev
        type(ReservoirCurve), intent(in) :: curve
        real(dp), intent(out) :: s_new, h_new, qout
        integer, intent(in), optional :: max_iter
        real(dp), intent(in), optional :: tol

        integer :: iter, iter_max
        real(dp) :: eps, q_old, q_new
        real(dp) :: s_trial, h_trial
        real(dp) :: smin, smax
        real(dp) :: available_release, excess_storage

        if (curve%n_sh < 2 .or. curve%n_hq < 2) then
            error stop "Reservoir curves must contain at least two points."
        end if

        if (.not. allocated(curve%storage) .or. .not. allocated(curve%level_sh) .or. &
            .not. allocated(curve%level_hq) .or. .not. allocated(curve%release)) then
            error stop "Reservoir curve arrays are not allocated."
        end if

        if (dt <= 0.0_dp) then
            error stop "Time step dt must be positive."
        end if

        iter_max = 100
        if (present(max_iter)) iter_max = max_iter

        eps = 1.0e-8_dp
        if (present(tol)) eps = tol

        smin = curve%storage(1)
        smax = curve%storage(curve%n_sh)

        ! 根据时步初始库容估计初始下泄流量。
        h_trial = interp_linear(curve%storage, curve%level_sh, curve%n_sh, &
                                min(max(s_prev, smin), smax))
        q_old = max(0.0_dp, interp_linear(curve%level_hq, curve%release, curve%n_hq, h_trial))

        ! 求解水量连续方程与 S-H、H-Q 关系的耦合过程。
        do iter = 1, iter_max

            s_trial = s_prev + (qin - q_old) * dt
            s_trial = min(max(s_trial, smin), smax)

            h_trial = interp_linear(curve%storage, curve%level_sh, curve%n_sh, s_trial)
            q_new = max(0.0_dp, interp_linear(curve%level_hq, curve%release, curve%n_hq, h_trial))

            if (abs(q_new - q_old) <= eps * max(1.0_dp, abs(q_old))) exit

            q_old = 0.5_dp * q_old + 0.5_dp * q_new
        end do

        qout = q_new
        s_new = s_prev + (qin - qout) * dt

        if (s_new > smax) then
            excess_storage = s_new - smax
            qout = qout + excess_storage / dt
            s_new = smax
        end if

        if (s_new < smin) then
            available_release = max(0.0_dp, (s_prev + qin * dt - smin) / dt)
            qout = min(qout, available_release)
            s_new = s_prev + (qin - qout) * dt
            s_new = max(s_new, smin)
        end if

        h_new = interp_linear(curve%storage, curve%level_sh, curve%n_sh, s_new)

    end subroutine reservoir_step


    function interp_linear(x, y, n, xq) result(yq)

        integer, intent(in) :: n
        real(dp), intent(in) :: x(n), y(n), xq
        real(dp) :: yq
        integer :: i
        real(dp) :: r

        if (xq <= x(1)) then
            yq = y(1)
            return
        end if

        if (xq >= x(n)) then
            yq = y(n)
            return
        end if

        do i = 1, n - 1
            if (xq >= x(i) .and. xq <= x(i + 1)) then
                if (abs(x(i + 1) - x(i)) < tiny(1.0_dp)) then
                    yq = y(i)
                else
                    r = (xq - x(i)) / (x(i + 1) - x(i))
                    yq = y(i) + r * (y(i + 1) - y(i))
                end if
                return
            end if
        end do

        error stop "Interpolation failed: check monotonic reservoir curve data."

    end function interp_linear

end module reservoir_regulation_module
