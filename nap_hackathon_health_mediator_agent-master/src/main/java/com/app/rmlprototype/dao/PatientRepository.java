package com.app.rmlprototype.dao;

import com.app.rmlprototype.entity.Patient;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PatientRepository extends JpaRepository<Patient,Integer> {

    @Query("SELECT p FROM Patient p WHERE UPPER(p.name) like %?1")
    public List<Patient> findByPatientName(String patientName);

    @Query("SELECT p FROM Patient p WHERE p.idProofNumber = ?1")
    public Patient findByPatientIdProof(String idProofNumber);

    @Query("SELECT p FROM Patient p WHERE p.idProofType like %?1")
    public List<Patient> findByIdProofType(String idProofType);

    @Query("delete from Patient p WHERE p.idProofNumber = ?1")
    public void deleteByPatientIdProof(String idProofNumber);

    @Query("SELECT p from Patient p WHERE p.username = ?1 and p.password = ?2")
    public Patient authenticateUser(String username,String password);

}
