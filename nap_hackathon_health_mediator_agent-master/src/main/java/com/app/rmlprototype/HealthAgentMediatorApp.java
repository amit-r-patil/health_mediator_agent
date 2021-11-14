package com.app.rmlprototype;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class HealthAgentMediatorApp {
    public static void main(String[] args) {
        SpringApplication.run(HealthAgentMediatorApp.class,args);
        System.out.println("App started");
    }
}
