package com.app.rmlprototype.rest;

import com.app.rmlprototype.data.MessageTemplates;
import com.app.rmlprototype.entity.DiagnosisCenter;
import com.app.rmlprototype.service.DiagnosisCenterSevice;
import com.app.rmlprototype.util.RapidAPIClient;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/diagnosiscenter")
public class DiagnosisCenterRestController {

    @Autowired
    private DiagnosisCenterSevice diagnosisCenterSevice;

    /*public DiagnosisCenterRestController(DiagnosisCenterSevice diagnosisCenterSevice){
        this.diagnosisCenterSevice = diagnosisCenterSevice;
    }*/

    @GetMapping("/authenticateCenter/{username}/{password}")
    public DiagnosisCenter authenticateUser(@PathVariable String username, @PathVariable String password){
        return diagnosisCenterSevice.authenticateUser(username,password);
    }

    @GetMapping("/getAllCenters")
    public List<DiagnosisCenter> getAllDoctors(){
        return diagnosisCenterSevice.findAll();
    }

    @GetMapping("/getCenter/{theId}")
    public DiagnosisCenter getSingleCenter(@PathVariable int theId){
        DiagnosisCenter diagnosisCenter = diagnosisCenterSevice.findById(theId);
        if(diagnosisCenter==null){
            throw new RuntimeException("Doctor Not Found");
        }
        return diagnosisCenter;
    }

    @GetMapping("/getCenter/{registrationNumber}")
    public DiagnosisCenter getCenterByRegNumber(@PathVariable String registrationNumber){
        DiagnosisCenter diagnosisCenter = diagnosisCenterSevice.findByRegistrationNumer(registrationNumber);
        if(diagnosisCenter==null){
            throw new RuntimeException("Doctor Not Found");
        }
        return diagnosisCenter;
    }

    @GetMapping("/getAllCenters/{name}")
    public List<DiagnosisCenter> getAllCentersByName(@PathVariable String name){
        return diagnosisCenterSevice.findByName(name);
    }

    @PostMapping("/addCenter")
    public DiagnosisCenter addDiagnosisCenter(@RequestBody DiagnosisCenter diagnosisCenter){
        diagnosisCenterSevice.save(diagnosisCenter);

        RapidAPIClient.sendSMS(diagnosisCenter.getContactNumber(), MessageTemplates.getDiagnosisCenterRegistartionSMS(diagnosisCenter.getName()));
        RapidAPIClient.sendEmail(diagnosisCenter.getEmail(), diagnosisCenter.getName(), MessageTemplates.REGISTRATION_SUCCESS_EMAIL_SUBJECT, MessageTemplates.getDiagnosisCenterRegistartionEmail(diagnosisCenter.getName()), null);

        return diagnosisCenter;
    }

    @DeleteMapping("/deleteCenterByRegistrationNumber/{registrationNumber}")
    public String delCenterByRegNum(@PathVariable String registrationNumber){
        DiagnosisCenter diagnosisCenter = diagnosisCenterSevice.findByRegistrationNumer(registrationNumber);
        if(diagnosisCenter==null){
            throw new RuntimeException("Doctor Not Found");
        }
        diagnosisCenterSevice.deleteByRegistrationNumber(registrationNumber);
        return "Center Deleted Successfully";
    }

    @DeleteMapping("/deleteCenterById/{theId}")
    public String delCenterById(@PathVariable int theId){
        DiagnosisCenter diagnosisCenter = diagnosisCenterSevice.findById(theId);
        if(diagnosisCenter==null){
            throw new RuntimeException("Doctor Not Found");
        }
        diagnosisCenterSevice.deleteById(theId);
        return "Center Deleted Successfully";
    }

}