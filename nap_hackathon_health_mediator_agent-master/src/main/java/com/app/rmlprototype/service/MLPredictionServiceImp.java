package com.app.rmlprototype.service;

import com.app.rmlprototype.dao.PredictionRepository;
import com.app.rmlprototype.entity.MLPrediction;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class MLPredictionServiceImp implements MLPredictionService{

    @Autowired
    private PredictionRepository predictionRepository;

    @Override
    public List<MLPrediction> findALl() {
        return predictionRepository.findAll();
    }

    @Override
    public List<MLPrediction> findDocumentBySpeciality(String speciality) {
        return predictionRepository.getAllBySpeciality(speciality);
    }

    @Override
    public MLPrediction getDocumentById(int id) {
        return predictionRepository.getReportById(id);
    }

    /*@Override
    public List<MLPrediction> getInfoByDocumentId(List<Integer> docIds) {
        return predictionRepository.getInfoByDocumentId(docIds);
    }*/

    @Override
    public void save(MLPrediction mlPrediction) {
        predictionRepository.save(mlPrediction);
    }
}
