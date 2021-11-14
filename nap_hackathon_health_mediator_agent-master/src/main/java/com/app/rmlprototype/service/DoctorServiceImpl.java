package com.app.rmlprototype.service;

import com.app.rmlprototype.dao.DoctorRepository;
import com.app.rmlprototype.entity.Doctor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class DoctorServiceImpl implements DoctorService{

    @Autowired
    private DoctorRepository doctorRepository;

    @Override
    public List<Doctor> findAll() {
        return doctorRepository.findAll();
    }

    @Override
    public Doctor findById(int theId) {
        Optional<Doctor> res = doctorRepository.findById(theId);
        Doctor theDoc = null;
        if (res.isPresent()) {
            theDoc = res.get();
        } else {
            throw new RuntimeException("Not Found");
        }

        return theDoc;
    }

    @Override
    public void save(Doctor theDoc) {
        doctorRepository.save(theDoc);
    }

    @Override
    public void deleteById(int theId) {
        doctorRepository.deleteById(theId);
    }

    /*@Override
    public Doctor findByDocId(String docId) {
        return doctorRepository.findByDocId(docId);
    }*/

    @Override
    public Doctor findByDocRegistrationNumber(String registrationNumber) {
        return doctorRepository.findByDocRegistrationNumber(registrationNumber);
    }

    @Override
    public List<Doctor> findByDocSpecialization(String specialization) {
        return doctorRepository.findByDocSpecialization(specialization);
    }

    @Override
    public void deleteByDocRegistrationNumber(String registrationNumber) {
        doctorRepository.deleteByDocRegistrationNumber(registrationNumber);
    }

    @Override
    public Doctor authenticateUser(String username, String password) {
        Doctor doctor = doctorRepository.authenticateUser(username,password);;

        return doctor;
    }
}
