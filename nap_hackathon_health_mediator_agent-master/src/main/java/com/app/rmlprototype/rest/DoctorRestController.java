package com.app.rmlprototype.rest;

import com.app.rmlprototype.data.MessageTemplates;
import com.app.rmlprototype.entity.Doctor;
import com.app.rmlprototype.service.DoctorService;
import com.app.rmlprototype.util.RapidAPIClient;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import javax.print.Doc;
import java.util.List;

@RestController
@RequestMapping("/api/doctor")
public class DoctorRestController {

    @Autowired
    private DoctorService doctorService;

    /*public DoctorRestController(DoctorService doctorService){
        this.doctorService = doctorService;
    }*/

    @GetMapping("/hello")
    public String sayHello(){
        return "Hello";
    }

    @GetMapping("/authenticateDoctor/{username}/{password}")
    public Doctor authenticateUser(@PathVariable String username, @PathVariable String password){
        return doctorService.authenticateUser(username,password);
    }

    @GetMapping("/getAllDoctors")
    public List<Doctor> getAllDoctors(){
        return doctorService.findAll();
    }

    @GetMapping("/getDoctor/{theId}")
    public Doctor getSingleDoc(@PathVariable int theId){
        Doctor doctor = doctorService.findById(theId);
        if(doctor==null){
            throw new RuntimeException("Doctor Not Found");
        }
        return doctor;
    }

    /*@GetMapping("/getDoctorByDocId/{theDoctorId}")
    public Doctor getByDocId(@PathVariable String theDoctorId){
        Doctor doctor = doctorService.findByDocId(theDoctorId);
        if(doctor==null){
            throw new RuntimeException("Doctor Not Found");
        }
        return doctor;
    }*/

    @GetMapping("/getDoctorByRegNum/{registrationNumber}")
    public Doctor getByDocRegistrationNumber(@PathVariable String registrationNumber){
        Doctor doctor = doctorService.findByDocRegistrationNumber(registrationNumber);
        if(doctor==null){
            throw new RuntimeException("Doctor Not Found");
        }
        return doctor;
    }

    @GetMapping("/getAllDoctorsBySpecialization/{specialization}")
    public List<Doctor> getAllDoctors(@PathVariable String specialization){
        return doctorService.findByDocSpecialization(specialization.toUpperCase());
    }

    @PostMapping("/addDoctor")
    public Doctor addDoc(@RequestBody Doctor doctor){
        doctorService.save(doctor);

        RapidAPIClient.sendSMS(doctor.getContactNumber(), MessageTemplates.getDoctorRegistartionSMS(doctor.getDocName()));
        RapidAPIClient.sendEmail(doctor.getEmail(), doctor.getDocName(), MessageTemplates.REGISTRATION_SUCCESS_EMAIL_SUBJECT, MessageTemplates.getDoctorRegistartionEmail(doctor.getDocName()), null);

        return doctor;
    }

    @DeleteMapping("/deleteDoctorById/{theId}")
    public String delDoctor(@PathVariable int theId){
        Doctor doctor = doctorService.findById(theId);
        if(doctor==null){
            throw new RuntimeException("Doctor Not Found");
        }
        doctorService.deleteById(theId);
        return "Doctor Deleted Successfully";
    }

    @DeleteMapping("/deleteDoctorByRegNumber/{registrationNumber}")
    public String delDoctor(@PathVariable String registrationNumber){
        Doctor doctor = doctorService.findByDocRegistrationNumber(registrationNumber);
        if(doctor==null){
            throw new RuntimeException("Doctor Not Found");
        }
        doctorService.deleteByDocRegistrationNumber(registrationNumber);
        return "Doctor Deleted Successfully";
    }

}