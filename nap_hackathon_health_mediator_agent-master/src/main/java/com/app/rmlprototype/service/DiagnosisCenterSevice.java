package com.app.rmlprototype.service;

import com.app.rmlprototype.entity.DiagnosisCenter;
import com.app.rmlprototype.entity.Patient;

import java.util.List;

public interface DiagnosisCenterSevice {
    public List<DiagnosisCenter> findAll();

    public DiagnosisCenter findById(int theId);

    public void save(DiagnosisCenter theEmployee);

    public void deleteById(int theId);

    public DiagnosisCenter findByRegistrationNumer(String registrationNumber);

    public List<DiagnosisCenter> findByName(String name);

    public void deleteByRegistrationNumber(String registrationNumber);

    public DiagnosisCenter authenticateUser(String username,String password);

}
