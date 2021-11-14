package com.app.rmlprototype.dao;

import com.app.rmlprototype.entity.MLPrediction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PredictionRepository extends JpaRepository<MLPrediction,Integer> {

    @Query("Select a from MLPrediction a where a.docId = ?1")
    public MLPrediction getReportById(int docId);

    @Query("SELECT a FROM MLPrediction a WHERE a.specialistNeeded like %?1")
    public List<MLPrediction> getAllBySpeciality(String speciality);

    /*@Query("Select a from MLPrediction a where a.docId in (:ids)")
    public List<MLPrediction> getInfoByDocumentId(List<Integer> ids);*/


}
