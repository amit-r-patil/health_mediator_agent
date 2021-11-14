package com.app.rmlprototype.dao;

import com.app.rmlprototype.entity.Doctor;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface DoctorRepository extends JpaRepository<Doctor,Integer> {

    /*@Query("SELECT d FROM Doctor d WHERE d.doctorId = ?1")
    public Doctor findByDocId(String doctorId);*/

    @Query("SELECT d FROM Doctor d WHERE d.registrationNumber = ?1")
    public Doctor findByDocRegistrationNumber(String registrationNumber);

    @Query("SELECT d FROM Doctor d WHERE d.specialization like %?1")
    public List<Doctor> findByDocSpecialization(String specialization);

    @Query("delete from Doctor d WHERE d.registrationNumber = ?1")
    public void deleteByDocRegistrationNumber(String registrationNumber);

    @Query("SELECT d from Doctor d WHERE d.username = ?1 and d.password = ?2")
    public Doctor authenticateUser(String username, String password);

}
