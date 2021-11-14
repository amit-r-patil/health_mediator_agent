package com.app.rmlprototype.service;

import com.app.rmlprototype.entity.Doctor;
import com.app.rmlprototype.entity.MLPrediction;

import java.util.List;

public interface MLPredictionService {
    public List<MLPrediction> findALl();

    public List<MLPrediction> findDocumentBySpeciality(String speciality);

    public MLPrediction getDocumentById(int id);

    //public List<MLPrediction> getInfoByDocumentId(List<Integer> docIds);

    public void save(MLPrediction mlPrediction);

}
