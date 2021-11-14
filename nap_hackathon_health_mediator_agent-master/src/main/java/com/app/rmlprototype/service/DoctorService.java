package com.app.rmlprototype.service;

import com.app.rmlprototype.entity.Doctor;

import javax.print.Doc;
import java.util.List;

public interface DoctorService {
    public List<Doctor> findAll();

    public Doctor findById(int theId);

    public void save(Doctor theDoc);

    public void deleteById(int theId);

   // public Doctor findByDocId(String docId);

    public Doctor findByDocRegistrationNumber(String registrationNumber);

    public List<Doctor> findByDocSpecialization(String specialization);

    public void deleteByDocRegistrationNumber(String registrationNumber);

    public Doctor authenticateUser(String username,String password);
}
