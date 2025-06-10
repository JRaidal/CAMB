program driver
    use CAMB
    use SpherBessels
    use config
    implicit none
    integer :: i

    do i = 1, 3
        call test()
    end do

    call Bessels_Free()
    deallocate(highL_CL_template)

contains

    subroutine test()
        implicit none
        type(CAMBdata) :: OutData
        !This is fine
        ! allocate(OutData%ClData%CTransScal%Delta_p_l_k(3, 1000, 800))
        !this is not
        allocate(OutData%ClData%CTransTens%Delta_p_l_k(3, 1000, 1000))
        OutData%ClData%CTransTens%Delta_p_l_k =1
        write(*,*) sum(OutData%ClData%CTransTens%Delta_p_l_k)

    end subroutine test

end program driver
