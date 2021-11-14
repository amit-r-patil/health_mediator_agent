package com.app.rmlprototype.data;

public class MessageTemplates {

    public static String REGISTRATION_SUCCESS_SMS = "Hello {{1}}, Your {{2}} profile has been created on Health Mediator Agent Website.";

    public static String REGISTRATION_SUCCESS_EMAIL_SUBJECT = "Registration Completed - Health Mediator Agent";

    public static String REGISTRATION_SUCCESS_EMAIL_MESSAGE = "Hello {{1}},\nYour {{2}} profile has been created  on Health Mediator Agent App. \nWarm Regards,\nHealth Mediator Agent Website";

    public static String REPORT_ADDED_SMS_PATIENT = "Hello, Your lab reports has been uploaded to Health Mediator Agent Website. You can check them right now.";

    public static String REPORT_ADDED_SMS_DOCTOR = "Hello Doctor, Lab reports of your patient {{1}} has been uploaded to Health Mediator Agent Website. You can check them right now.";

    public static String REPORT_ADDED_EMAIL_SUBJECT_PATIENT = "Health Mediator Agent - Check Your lab Reports";

    public static String REPORT_ADDED_EMAIL_SUBJECT_DOCTOR = "Health Mediator Agent - Check Your Patien's lab Reports";

    public static String REPORT_ADDED_EMAIL_MESSAGE_PATIENT = "Hello {{1}}, \nPlease find attached lab reports. You can also check same on Health Mediator Agent Website or your registered whats app number. \nWarm Regards,\nHealth Mediator Agent Website";

    public static String REPORT_ADDED_EMAIL_MESSAGE_DOCTOR = "Hello Doctor, Please find attached lab reports of your patient {{1}}. You can also check same on Health Mediator Agent Website or your registered whats app number. \nWarm Regards,\nHealth Mediator Agent Website";

    public static String getPatientRegistartionSMS(String username) {
        return REGISTRATION_SUCCESS_SMS.replace("{{1}}", username).replace("{{2}}", "Patient");

    }

    public static String getDoctorRegistartionSMS(String username) {
        return REGISTRATION_SUCCESS_SMS.replace("{{1}}", username).replace("{{2}}", "Doctor");

    }

    public static String getDiagnosisCenterRegistartionSMS(String username) {
        return REGISTRATION_SUCCESS_SMS.replace("{{1}}", username).replace("{{2}}", "Diagnosis Center");

    }

    public static String getPatientRegistartionEmail(String username) {
        return REGISTRATION_SUCCESS_EMAIL_MESSAGE.replace("{{1}}", username).replace("{{2}}", "Patient");

    }

    public static String getDoctorRegistartionEmail(String username) {
        return REGISTRATION_SUCCESS_SMS.replace("{{1}}", username).replace("{{2}}", "Doctor");

    }

    public static String getDiagnosisCenterRegistartionEmail(String username) {
        return REGISTRATION_SUCCESS_EMAIL_MESSAGE.replace("{{1}}", username).replace("{{2}}", "Diagnosis Center");

    }

    public static String getReportAddedSMSDoctor(String patientname) {
        return REGISTRATION_SUCCESS_EMAIL_MESSAGE.replace("{{1}}", patientname);
    }

    public static String getReportAddedEmailForPatient(String patientname) {
        return REPORT_ADDED_EMAIL_MESSAGE_PATIENT.replace("{{1}}", patientname);
    }

    public static String getReportAddedEmailForDoctor(String patientname) {
        return REPORT_ADDED_EMAIL_MESSAGE_DOCTOR.replace("{{1}}", patientname);
    }
}