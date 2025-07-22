module ActiveSources
    use precision
    use classes
    implicit none

    private
    public :: TActiveSources, TActiveSources_SetCorrelatorTable, TActiveSources_SetActiveEigenmode, &
              TActiveSources_GetScalarSources, TActiveSources_GetVectorSources, TActiveSources_GetTensorSources

    type, extends(TPythonInterfacedClass) :: TActiveSources

        ! Grid and raw data tables
        real(dl), allocatable    :: k_grid(:)
        real(dl), allocatable    :: tau_grid(:)
        integer                  :: nk = 0
        integer                  :: ntau = 0

        real(dl), allocatable    :: eigenfunctions_table(:,:,:,:)
        real(dl), allocatable    :: eigenfunc_derivs_table(:,:,:,:)

        ! Eigenvalues
        real(dl), allocatable    :: eigenvalues_S(:,:)
        real(dl), allocatable    :: eigenvalues_00(:,:)
        real(dl), allocatable    :: eigenvalues_V(:,:)
        real(dl), allocatable    :: eigenvalues_T(:,:)

        logical                  :: tables_are_set = .false.

        integer                  :: nmodes = 0
        integer                  :: ntypes = 0
        real(dl)                 :: string_mu = 0.0_dl
        real(dl)                 :: weighting = 0.25_dl ! Scaling parameter for source terms

        integer                  :: active_mode_idx = 0 ! 0 means no active source, 1 to N for modes

    contains
        procedure, nopass :: PythonClass => TActiveSources_PythonClass
        procedure, nopass :: SelfPointer => TActiveSources_SelfPointer
        procedure :: SetCorrelatorTable => TActiveSources_SetCorrelatorTable
        procedure :: SetActiveEigenmode => TActiveSources_SetActiveEigenmode
        procedure :: GetScalarSources => TActiveSources_GetScalarSources
        procedure :: GetVectorSources => TActiveSources_GetVectorSources
        procedure :: GetTensorSources => TActiveSources_GetTensorSources
    end type TActiveSources

contains

    ! ---------------------------------------------------------------------------------
    ! CORE ROUTINES
    ! ---------------------------------------------------------------------------------

    subroutine TActiveSources_SetActiveEigenmode(this, mode_to_set)
        class(TActiveSources), intent(inout) :: this
        integer, intent(in)           :: mode_to_set

        if (mode_to_set >= 0 .and. (mode_to_set == 0 .or. mode_to_set <= this%nmodes) ) then
            this%active_mode_idx = mode_to_set
            if (mode_to_set > 0) then
                print *, "Fortran TActiveSources: Active UETC eigenmode set to: ", this%active_mode_idx
            else
                print *, "Fortran TActiveSources: UETC sources turned OFF (active_mode_idx = 0)."
            endif
        else
            print *, "Fortran TActiveSources Warning: Invalid active_mode_idx requested: ", mode_to_set, &
                     " Max modes available: ", this%nmodes, ". Setting to 0 (OFF)."
            this%active_mode_idx = 0
        endif
    end subroutine TActiveSources_SetActiveEigenmode

    subroutine TActiveSources_GetScalarSources(this, k_val, tau_val, emt00, emtS, emt00dot, emtSdot)
        class(TActiveSources), intent(in) :: this
        real(dl), intent(in) :: k_val, tau_val
        real(dl), intent(out) :: emt00, emtS, emt00dot, emtSdot

        real(dl) :: ktau, common_scaling_corr, eigenvalue_k
        real(dl) :: u_00, u_S
        real(dl) :: du00_dlogkt, duS_dlogkt, dot_numerator_00, dot_numerator_S
        real(dl) :: dot_terms_den
        integer :: eigenmode

        emt00 = 0.0_dl; emtS = 0.0_dl; emt00dot = 0.0_dl; emtSdot = 0.0_dl

        if (this%active_mode_idx <= 0 .or. .not. this%tables_are_set) return
        eigenmode = this%active_mode_idx
        ktau = k_val * tau_val

        ! Perform interpolation directly from raw tables
        u_00 = BilinearInterp(k_val, ktau, this%k_grid, this%tau_grid, this%eigenfunctions_table(:, 1, eigenmode, :))
        u_S  = BilinearInterp(k_val, ktau, this%k_grid, this%tau_grid, this%eigenfunctions_table(:, 2, eigenmode, :))
        eigenvalue_k = LinearInterp(k_val, this%k_grid, this%eigenvalues_S(:, eigenmode))

        common_scaling_corr = sqrt(abs(eigenvalue_k)) / ((ktau**this%weighting) * sqrt(tau_val))
        emt00 = u_00 * common_scaling_corr
        emtS = u_S * common_scaling_corr

        ! Calculate derivatives using bilinear interpolation
        du00_dlogkt = BilinearInterp(k_val, ktau, this%k_grid, this%tau_grid, this%eigenfunc_derivs_table(:, 1, eigenmode, :))
        duS_dlogkt  = BilinearInterp(k_val, ktau, this%k_grid, this%tau_grid, this%eigenfunc_derivs_table(:, 2, eigenmode, :))

        dot_numerator_00 = du00_dlogkt - (this%weighting + 0.5_dl) * u_00
        dot_numerator_S  = duS_dlogkt - (this%weighting + 0.5_dl) * u_S

        dot_terms_den = tau_val
        if (abs(dot_terms_den) > 1.0d-150) then
            emt00dot = dot_numerator_00 * common_scaling_corr / dot_terms_den
            emtSdot  = dot_numerator_S  * common_scaling_corr / dot_terms_den
        endif

    end subroutine TActiveSources_GetScalarSources

    subroutine TActiveSources_GetVectorSources(this, k_val, tau_val, emtV)
        class(TActiveSources), intent(in) :: this
        real(dl), intent(in) :: k_val, tau_val
        real(dl), intent(out) :: emtV

        real(dl) :: ktau, common_scaling_corr, eigenvalue_k
        integer :: eigenmode
        emtV = 0.0_dl

        if (this%active_mode_idx <= 0 .or. .not. this%tables_are_set) return
        eigenmode = this%active_mode_idx
        ktau = k_val * tau_val

        eigenvalue_k = LinearInterp(k_val, this%k_grid, this%eigenvalues_V(:, eigenmode))
        common_scaling_corr = sqrt(abs(eigenvalue_k)) / ((ktau**this%weighting) * sqrt(tau_val))
        emtV = BilinearInterp(k_val, ktau, this%k_grid, this%tau_grid, this%eigenfunctions_table(:, 3, eigenmode, :)) * common_scaling_corr

    end subroutine TActiveSources_GetVectorSources

    subroutine TActiveSources_GetTensorSources(this, k_val, tau_val, emtT)
        class(TActiveSources), intent(in) :: this
        real(dl), intent(in) :: k_val, tau_val
        real(dl), intent(out) :: emtT

        real(dl) :: ktau, common_scaling_corr, eigenvalue_k
        integer :: eigenmode
        emtT = 0.0_dl

        if (this%active_mode_idx <= 0 .or. .not. this%tables_are_set) return
        eigenmode = this%active_mode_idx
        ktau = k_val * tau_val

        eigenvalue_k = LinearInterp(k_val, this%k_grid, this%eigenvalues_T(:, eigenmode))
        common_scaling_corr = sqrt(abs(eigenvalue_k)) / ((ktau**this%weighting) * sqrt(tau_val))
        emtT = BilinearInterp(k_val, ktau, this%k_grid, this%tau_grid, this%eigenfunctions_table(:, 4, eigenmode, :)) * common_scaling_corr

    end subroutine TActiveSources_GetTensorSources

    ! ---------------------------------------------------------------------------------
    ! DATA LOADING AND SETUP
    ! ---------------------------------------------------------------------------------

    subroutine TActiveSources_SetCorrelatorTable(this, &
        k_grid_in, nk_in, &
        tau_grid_in, ntau_in, &
        eigenfuncs_flat_in, num_eigen_types_in, nmodes_in, &
        evals_S_flat_in, evals_00_flat_in, evals_V_flat_in, evals_T_flat_in, &
        eigenfunc_derivs_logkt_flat_in, &
        mu_in, weighting_param_in)

        class(TActiveSources), intent(inout) :: this
        integer, intent(in)                :: nk_in, ntau_in, num_eigen_types_in, nmodes_in
        real(dl), intent(in)               :: k_grid_in(nk_in), tau_grid_in(ntau_in)
        real(dl), intent(in)               :: eigenfuncs_flat_in(*)
        real(dl), intent(in)               :: eigenfunc_derivs_logkt_flat_in(*)
        real(dl), intent(in)               :: evals_S_flat_in(*),evals_00_flat_in(*), evals_V_flat_in(*), evals_T_flat_in(*)
        real(dl), intent(in)               :: mu_in, weighting_param_in

        integer :: i, j, l, m, idx_flat

        call DeallocateCorrelatorTable(this)
        this%tables_are_set = .false.

        this%nk = nk_in; this%ntau = ntau_in
        this%nmodes = nmodes_in; this%ntypes = num_eigen_types_in
        this%string_mu = mu_in; this%weighting = weighting_param_in

        print *, "Fortran TActiveSources_SetCorrelatorTable: Received nk=", this%nk, ", ntau=", this%ntau, &
                 ", ntypes=", this%ntypes, ", nmodes=", this%nmodes

        if (this%nk < 2 .or. this%ntau < 2 .or. this%nmodes <= 0 .or. this%ntypes /= 4) then
            print *, "Fortran SetCorrelatorTable: Insufficient dimensions. Aborting setup."
            call EnsureZeroSizeAllocations(this)
            return
        endif

        allocate(this%k_grid(this%nk)); this%k_grid = k_grid_in
        allocate(this%tau_grid(this%ntau)); this%tau_grid = tau_grid_in

        ! --- THIS IS THE CORRECTED SECTION ---
        ! Manually reshape the flat arrays into the multi-dimensional tables
        allocate(this%eigenfunctions_table(this%nk, this%ntypes, this%nmodes, this%ntau))
        allocate(this%eigenfunc_derivs_table(this%nk, this%ntypes, this%nmodes, this%ntau))

        do i = 1, this%nk      ! k_index
            do j = 1, this%ntypes  ! type_index
                do l = 1, this%nmodes  ! mode_index
                    do m = 1, this%ntau ! tau_index
                        ! This indexing assumes Fortran-style ('F') ordering from the calling language (e.g. numpy.flatten(order='F'))
                        idx_flat = ((((i-1) * this%ntypes + (j-1)) * this%nmodes + (l-1)) * this%ntau + m-1) + 1
                        this%eigenfunctions_table(i, j, l, m) = eigenfuncs_flat_in(idx_flat)
                        this%eigenfunc_derivs_table(i, j, l, m) = eigenfunc_derivs_logkt_flat_in(idx_flat)
                    end do
                end do
            end do
        end do
        ! --- END OF CORRECTION ---

        ! Store raw eigenvalues
        allocate(this%eigenvalues_S(this%nk, this%nmodes))
        allocate(this%eigenvalues_00(this%nk, this%nmodes))
        allocate(this%eigenvalues_V(this%nk, this%nmodes))
        allocate(this%eigenvalues_T(this%nk, this%nmodes))

        this%eigenvalues_S  = transpose(reshape(evals_S_flat_in(1:this%nk*this%nmodes), [this%nmodes, this%nk]))
        this%eigenvalues_00 = transpose(reshape(evals_00_flat_in(1:this%nk*this%nmodes), [this%nmodes, this%nk]))
        this%eigenvalues_V  = transpose(reshape(evals_V_flat_in(1:this%nk*this%nmodes), [this%nmodes, this%nk]))
        this%eigenvalues_T  = transpose(reshape(evals_T_flat_in(1:this%nk*this%nmodes), [this%nmodes, this%nk]))

        print *, "Fortran SetCorrelatorTable: Raw data tables (functions, derivatives, eigenvalues) stored."
        this%tables_are_set = .true.

    contains
        subroutine DeallocateCorrelatorTable(obj_internal)
            class(TActiveSources), intent(inout) :: obj_internal
            if (allocated(obj_internal%k_grid)) deallocate(obj_internal%k_grid)
            if (allocated(obj_internal%tau_grid)) deallocate(obj_internal%tau_grid)
            if (allocated(obj_internal%eigenfunctions_table)) deallocate(obj_internal%eigenfunctions_table)
            if (allocated(obj_internal%eigenfunc_derivs_table)) deallocate(obj_internal%eigenfunc_derivs_table)
            if (allocated(obj_internal%eigenvalues_S)) deallocate(obj_internal%eigenvalues_S)
            if (allocated(obj_internal%eigenvalues_00)) deallocate(obj_internal%eigenvalues_00)
            if (allocated(obj_internal%eigenvalues_V)) deallocate(obj_internal%eigenvalues_V)
            if (allocated(obj_internal%eigenvalues_T)) deallocate(obj_internal%eigenvalues_T)
            obj_internal%nk = 0; obj_internal%ntau = 0;
            obj_internal%ntypes = 0; obj_internal%nmodes = 0
            obj_internal%tables_are_set = .false.
        end subroutine DeallocateCorrelatorTable

        subroutine EnsureZeroSizeAllocations(obj_internal)
             class(TActiveSources), intent(inout) :: obj_internal
             if (.not. allocated(obj_internal%k_grid)) allocate(obj_internal%k_grid(0))
             if (.not. allocated(obj_internal%tau_grid)) allocate(obj_internal%tau_grid(0))
             if (.not. allocated(obj_internal%eigenfunctions_table)) allocate(obj_internal%eigenfunctions_table(0,0,0,0))
             if (.not. allocated(obj_internal%eigenfunc_derivs_table)) allocate(obj_internal%eigenfunc_derivs_table(0,0,0,0))
             if (.not. allocated(obj_internal%eigenvalues_S)) allocate(obj_internal%eigenvalues_S(0,0))
             if (.not. allocated(obj_internal%eigenvalues_00)) allocate(obj_internal%eigenvalues_00(0,0))
             if (.not. allocated(obj_internal%eigenvalues_V)) allocate(obj_internal%eigenvalues_V(0,0))
             if (.not. allocated(obj_internal%eigenvalues_T)) allocate(obj_internal%eigenvalues_T(0,0))
        end subroutine EnsureZeroSizeAllocations
    end subroutine TActiveSources_SetCorrelatorTable

    ! ---------------------------------------------------------------------------------
    ! NEW INTERPOLATION FUNCTIONS
    ! ---------------------------------------------------------------------------------

    ! Performs fast 2D bilinear interpolation on a pre-loaded grid.
    function BilinearInterp(x, y, x_grid, y_grid, grid_data) result(val)
        implicit none
        real(dl), intent(in) :: x, y
        real(dl), intent(in) :: x_grid(:), y_grid(:)
        real(dl), intent(in) :: grid_data(:,:)
        real(dl)             :: val

        integer :: i, j
        real(dl) :: t, u
        real(dl) :: f_i_j, f_ip1_j, f_i_jp1, f_ip1_jp1

        i = find_lower_bound(x_grid, x)
        j = find_lower_bound(y_grid, y)

        t = (x - x_grid(i)) / (x_grid(i+1) - x_grid(i))
        u = (y - y_grid(j)) / (y_grid(j+1) - y_grid(j))

        f_i_j     = grid_data(i,   j)
        f_ip1_j   = grid_data(i+1, j)
        f_i_jp1   = grid_data(i,   j+1)
        f_ip1_jp1 = grid_data(i+1, j+1)

        val = f_i_j * (1.0_dl - t) * (1.0_dl - u) + &
              f_ip1_j * t * (1.0_dl - u) + &
              f_i_jp1 * (1.0_dl - t) * u + &
              f_ip1_jp1 * t * u
    end function BilinearInterp

    ! Performs fast 1D linear interpolation.
    function LinearInterp(x, x_grid, y_values) result(val)
        implicit none
        real(dl), intent(in) :: x
        real(dl), intent(in) :: x_grid(:), y_values(:)
        real(dl)             :: val

        integer :: i
        real(dl) :: t

        i = find_lower_bound(x_grid, x)

        t = (x - x_grid(i)) / (x_grid(i+1) - x_grid(i))

        val = y_values(i) * (1.0_dl - t) + y_values(i+1) * t
    end function LinearInterp

    ! Efficiently finds the lower index of a value in a sorted array.
    function find_lower_bound(arr, val) result(idx)
        implicit none
        real(dl), intent(in) :: arr(:)
        real(dl), intent(in) :: val
        integer              :: idx

        integer :: n, low, high, mid
        n = size(arr)

        if (val <= arr(1)) then
            idx = 1
            return
        endif
        if (val >= arr(n-1)) then
            idx = n - 1
            return
        endif

        low = 1
        high = n
        do while (low < high)
            mid = low + (high - low) / 2
            if (arr(mid) < val) then
                low = mid + 1
            else
                high = mid
            end if
        end do
        idx = low - 1
    end function find_lower_bound


    ! ---------------------------------------------------------------------------------
    ! PYTHON INTERFACE (No changes needed here)
    ! ---------------------------------------------------------------------------------

    function TActiveSources_PythonClass() result(pyClassName)
        character(LEN=:), allocatable :: pyClassName; pyClassName = 'ActiveSources'
    end function TActiveSources_PythonClass

    subroutine TActiveSources_SelfPointer(cptr, P)
        use iso_c_binding; Type(C_PTR) :: cptr
        class(TPythonInterfacedClass), pointer :: P; Type(TActiveSources), pointer :: P_Specific_TActiveSources
        call c_f_pointer(cptr, P_Specific_TActiveSources); P => P_Specific_TActiveSources
    end subroutine TActiveSources_SelfPointer

end module ActiveSources