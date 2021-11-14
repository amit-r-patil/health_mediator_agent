package com.app.rmlprototype.entity;

import javax.persistence.*;

@Entity
@Table(name = "ml_predictions")
public class MLPrediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "prediction_id")
    private int predictionId;

    @Column(name = "document_id")
    private int docId;

    @Column(name = "predictions")
    private String prediction;

    @Column(name = "specialist_needed")
    private String specialistNeeded;

    public int getPredictionId() {
        return predictionId;
    }

    public void setPredictionId(int predictionId) {
        this.predictionId = predictionId;
    }

    public int getDocId() {
        return docId;
    }

    public void setDocId(int docId) {
        this.docId = docId;
    }

    public String getPrediction() {
        return prediction;
    }

    public void setPrediction(String prediction) {
        this.prediction = prediction;
    }

    public String getSpecialistNeeded() {
        return specialistNeeded;
    }

    public void setSpecialistNeeded(String specialistNeeded) {
        this.specialistNeeded = specialistNeeded;
    }

    @Override
    public String toString() {
        return "MLPrediction{" +
                "predictionId=" + predictionId +
                ", docId=" + docId +
                ", prediction='" + prediction + '\'' +
                ", specialistNeeded='" + specialistNeeded + '\'' +
                '}';
    }
}
