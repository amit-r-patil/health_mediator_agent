package com.app.rmlprototype.service;

import com.app.rmlprototype.entity.Patient;

import java.util.List;

public interface PatientService {
    public List<Patient> findAll();

    public Patient findById(int theId);

    public void save(Patient theEmployee);

    public void deleteById(int theId);

    public List<Patient> findByPatientName(String patientName);

    public Patient findByPatientIdProof(String idProofNumber);

    public List<Patient> findByIdProofType(String idProofType);

    public void deleteByPatientIdProof(String idProofNumber);

    public Patient authenticateUser(String username,String password);
}
