package com.app.rmlprototype.service;

import com.app.rmlprototype.dao.PatientRepository;
import com.app.rmlprototype.entity.DiagnosisCenter;
import com.app.rmlprototype.entity.Patient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class PatientServiceImpl implements PatientService{

    @Autowired
    private PatientRepository patientRepository;

    @Override
    public List<Patient> findAll() {
        return patientRepository.findAll();
    }

    @Override
    public Patient findById(int theId) {
        Optional<Patient> res = patientRepository.findById(theId);
        Patient thePatient = null;
        if (res.isPresent()) {
            thePatient = res.get();
        } else {
            throw new RuntimeException("Not Found");
        }

        return thePatient;
    }

    @Override
    public void save(Patient patient) {
        patientRepository.save(patient);
    }

    @Override
    public void deleteById(int theId) {
        patientRepository.deleteById(theId);
    }

    @Override
    public List<Patient> findByPatientName(String patientName) {
        return patientRepository.findByPatientName(patientName);
    }

    @Override
    public Patient findByPatientIdProof(String idProofNumber) {
        return patientRepository.findByPatientIdProof(idProofNumber);
    }

    @Override
    public List<Patient> findByIdProofType(String idProofType) {
        return patientRepository.findByIdProofType(idProofType);
    }

    @Override
    public void deleteByPatientIdProof(String idProofNumber) {
        patientRepository.deleteByPatientIdProof(idProofNumber);
    }

    @Override
    public Patient authenticateUser(String username, String password) {
        Patient patient = patientRepository.authenticateUser(username,password);;

        return patient;
    }
}
