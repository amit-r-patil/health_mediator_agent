package com.app.rmlprototype.service;

import com.app.rmlprototype.dao.DocumentStoragePropertiesRepo;
import com.app.rmlprototype.entity.DocumnentStorageProperties;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.UrlResource;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.List;

import org.springframework.core.io.Resource;

@Service
public class DocumentStorageService {

    private Path fileStorageLocation;
    @Autowired
    private DocumentStoragePropertiesRepo docStorageRepo;

    private DocumnentStorageProperties documnentStorageProperties;

    public String storeFile(MultipartFile file, Integer userId, String docType) {
        this.fileStorageLocation = Paths.get(System.getProperty("user.home"))
                .toAbsolutePath().normalize();
        try {
            Files.createDirectories(this.fileStorageLocation);
        } catch (Exception ex) {
            throw new RuntimeException("Could not create the directory where the uploaded files will be stored.", ex);
        }

        // Normalize file name
        String originalFileName = StringUtils.cleanPath(file.getOriginalFilename());
        String fileName = "";
        try {
            // Check if the file's name contains invalid characters
            if(originalFileName.contains("..")) {
                throw new RuntimeException("Sorry! Filename contains invalid path sequence " + originalFileName);
            }
            String fileExtension = "";
            try {
                fileExtension = originalFileName.substring(originalFileName.lastIndexOf("."));
            } catch(Exception e) {
                fileExtension = "";
            }
            fileName = userId + "_" + docType + fileExtension;
            // Copy file to the target location (Replacing existing file with the same name)
            Path targetLocation = this.fileStorageLocation.resolve(fileName);
            Files.copy(file.getInputStream(), targetLocation, StandardCopyOption.REPLACE_EXISTING);
            DocumnentStorageProperties doc = docStorageRepo.checkDocumentByUserId(userId, docType);
            if(doc != null) {
                doc.setDocumentFormat(file.getContentType());
                doc.setFileName(fileName);
                docStorageRepo.save(doc);
            } else {
                DocumnentStorageProperties newDoc = new DocumnentStorageProperties();
                newDoc.setUserId(userId);
                newDoc.setDocumentFormat(file.getContentType());
                newDoc.setFileName(fileName);
                newDoc.setDocumentType(docType);
                docStorageRepo.save(newDoc);
            }
            return fileName;
        } catch (IOException ex) {
            throw new RuntimeException("Could not store file " + fileName + ". Please try again!", ex);

        }
    }
    public Resource loadFileAsResource(String fileName) throws Exception {

        this.fileStorageLocation = Paths.get(System.getProperty("user.home"))
                .toAbsolutePath().normalize();
        try {
            Files.createDirectories(this.fileStorageLocation);
        } catch (Exception ex) {
            throw new RuntimeException("Could not create the directory where the uploaded files will be stored.", ex);
        }

        try {
            Path filePath = this.fileStorageLocation.resolve(fileName).normalize();
            Resource resource = new UrlResource(filePath.toUri());
            if(resource.exists()) {
                return resource;
            } else {
                throw new FileNotFoundException("File not found " + fileName);
            }
        } catch (MalformedURLException ex) {
            throw new FileNotFoundException("File not found " + fileName);
        }
    }

    public String getDocumentName(Integer userId, String docType) {
        return docStorageRepo.getUploadDocumnetPath(userId, docType);
    }

    public DocumnentStorageProperties getDocumentOnId(int id){
        return docStorageRepo.getOne(id);
    }

    public List<DocumnentStorageProperties> getAllDocumentById(int id){
        return docStorageRepo.getDocumentByUserId(id);
    }

    public List<DocumnentStorageProperties> getAllDocument() {
        return docStorageRepo.findAll();
    }

}