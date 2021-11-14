package com.app.rmlprototype.dao;

import com.app.rmlprototype.entity.DiagnosisCenter;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface DiagnosisRepository extends JpaRepository<DiagnosisCenter,Integer> {
    @Query("SELECT d FROM DiagnosisCenter d WHERE d.registrationNumber = ?1")
    public DiagnosisCenter findByRegistrationNumer(String registrationNumber);

    @Query("SELECT d FROM DiagnosisCenter d WHERE d.name like %?1")
    public List<DiagnosisCenter> findByName(String name);

    @Query("delete from DiagnosisCenter d WHERE d.registrationNumber = ?1")
    public void deleteByRegistrationNumber(String registrationNumber);

    @Query("SELECT d from DiagnosisCenter d WHERE d.username = ?1 and d.password = ?2")
    public DiagnosisCenter authenticateUser(String username, String password);

}
