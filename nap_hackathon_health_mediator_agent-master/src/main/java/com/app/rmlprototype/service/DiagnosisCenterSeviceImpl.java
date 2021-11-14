package com.app.rmlprototype.service;

import com.app.rmlprototype.dao.DiagnosisRepository;
import com.app.rmlprototype.entity.DiagnosisCenter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class DiagnosisCenterSeviceImpl implements DiagnosisCenterSevice {

    @Autowired
    private DiagnosisRepository diagnosisRepository;

    @Override
    public List<DiagnosisCenter> findAll() {
        return diagnosisRepository.findAll();
    }

    @Override
    public DiagnosisCenter findById(int theId) {
        Optional<DiagnosisCenter> res = diagnosisRepository.findById(theId);
        DiagnosisCenter theDiagnosisCenter = null;
        if (res.isPresent()) {
            theDiagnosisCenter = res.get();
        } else {
            throw new RuntimeException("Not Found");
        }

        return theDiagnosisCenter;
    }

    @Override
    public void save(DiagnosisCenter diagnosisCenter) {
        diagnosisRepository.save(diagnosisCenter);
    }

    @Override
    public void deleteById(int theId) {
        diagnosisRepository.deleteById(theId);
    }

    @Override
    public DiagnosisCenter findByRegistrationNumer(String registrationNumber) {
        return diagnosisRepository.findByRegistrationNumer(registrationNumber);
    }

    @Override
    public List<DiagnosisCenter> findByName(String name) {
        return diagnosisRepository.findByName(name);
    }

    @Override
    public void deleteByRegistrationNumber(String registrationNumber) {
        diagnosisRepository.deleteByRegistrationNumber(registrationNumber);
    }

    @Override
    public DiagnosisCenter authenticateUser(String username, String password) {
        DiagnosisCenter theDiagnosisCenter = diagnosisRepository.authenticateUser(username,password);;

        return theDiagnosisCenter;
    }
}
