! CAMB-style memory leak test - complex nested types
module camb_test_types
    implicit none
    
    integer, parameter :: dl = kind(1.0d0)
    
    Type :: TCAMBdata
    end type TCAMBdata

    Type ClTransferData
        integer :: NumSources    
        real(dl), dimension(:,:,:), allocatable :: Delta_p_l_k
        integer :: max_index_nonlimber
    end Type ClTransferData
    
    type TCLdata
        Type(ClTransferData) :: CTransScal, CTransTens, CTransVec
        real(dl), dimension (:,:), allocatable :: Cl_scalar, Cl_tensor, Cl_vector
        real(dl), dimension (:,:,:), allocatable :: Cl_Scalar_Array
        integer :: lmax_lensed
        real(dl) , dimension (:,:), allocatable :: Cl_lensed
    end type    
    
    type, extends(TCAMBdata) :: CAMBdata   
        Type(TClData) :: CLdata
    end type

contains

    subroutine test_camb_memory()
        type(CAMBdata) :: P
        
        ! Allocate CTransScal and CTransTens (your specific concern)
        P%CLdata%CTransScal%NumSources = 3
        allocate(P%CLdata%CTransScal%Delta_p_l_k(100, 50, 3))
        P%CLdata%CTransScal%Delta_p_l_k = 1.0_dl
        
        P%CLdata%CTransTens%NumSources = 2
        allocate(P%CLdata%CTransTens%Delta_p_l_k(80, 40, 2))
        P%CLdata%CTransTens%Delta_p_l_k = 2.0_dl
        
        ! Allocate other arrays
        allocate(P%CLdata%Cl_scalar(200, 4))
        allocate(P%CLdata%Cl_tensor(150, 3))
        allocate(P%CLdata%Cl_Scalar_Array(300, 5, 6))
        allocate(P%CLdata%Cl_lensed(250, 4))
        
        P%CLdata%Cl_scalar = 3.0_dl
        P%CLdata%Cl_tensor = 4.0_dl
        P%CLdata%Cl_Scalar_Array = 5.0_dl
        P%CLdata%Cl_lensed = 6.0_dl
        
        ! P goes out of scope - should auto-deallocate everything
    end subroutine

    subroutine test_assignment()
        type(CAMBdata) :: P1, P2
        
        ! Allocate P1
        P1%CLdata%CTransScal%NumSources = 4
        allocate(P1%CLdata%CTransScal%Delta_p_l_k(120, 60, 4))
        P1%CLdata%CTransScal%Delta_p_l_k = 10.0_dl
        
        P1%CLdata%CTransTens%NumSources = 3
        allocate(P1%CLdata%CTransTens%Delta_p_l_k(90, 45, 3))
        P1%CLdata%CTransTens%Delta_p_l_k = 20.0_dl
        
        allocate(P1%CLdata%Cl_scalar(180, 5))
        P1%CLdata%Cl_scalar = 30.0_dl
        
        ! Assignment operation
        P2 = P1
        
        ! Modify P2 to ensure independence
        if (allocated(P2%CLdata%CTransScal%Delta_p_l_k)) then
            P2%CLdata%CTransScal%Delta_p_l_k = 100.0_dl
        end if
        
        ! Both go out of scope
    end subroutine

end module camb_test_types

program camb_memory_test
    use camb_test_types
    implicit none
    
    call test_camb_memory()
    call test_assignment()
    
end program camb_memory_test
