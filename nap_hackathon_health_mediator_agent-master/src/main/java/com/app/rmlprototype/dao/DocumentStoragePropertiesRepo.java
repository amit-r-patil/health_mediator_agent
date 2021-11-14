package com.app.rmlprototype.dao;

import com.app.rmlprototype.entity.DocumnentStorageProperties;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface DocumentStoragePropertiesRepo extends JpaRepository<DocumnentStorageProperties, Integer> {

    @Query("Select a from DocumnentStorageProperties a where a.UserId = ?1 and a.documentType = ?2")
    DocumnentStorageProperties checkDocumentByUserId(Integer userId, String docType);

    @Query("Select fileName from DocumnentStorageProperties a where a.UserId = ?1 and a.documentType = ?2")
    String getUploadDocumnetPath(Integer userId, String docType);

    @Query("Select a from DocumnentStorageProperties a where a.UserId = ?1")
    List<DocumnentStorageProperties> getDocumentByUserId(Integer userId);

}