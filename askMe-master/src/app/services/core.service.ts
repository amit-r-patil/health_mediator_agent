
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';


@Injectable({
  providedIn: 'root'
})
export class CoreService {

  constructor(private http: HttpClient) { }

  public getUser(): Observable<any>{
    return this.http.get('assets/users.json');
  }

  public getAllDoctors(){
    return this.http.get('/api/doctor/getAllDoctors');
  }

  public getAllPatients(){
    return this.http.get('/api/patient/getAllPatients');
  }

  public authenticateDoctor(userName, password){
    const url = '/api/doctor/authenticateDoctor/' + userName + '/' + password;
    return this.http.get(url);
  }

  public authenticateCenter(userName, password){
    const url = '/api/diagnosiscenter/authenticateCenter/' + userName + '/' + password;
    return this.http.get(url);
  }

  public authenticateUser(userName, password){
    const url = '/api/patient/authenticatePatient/' + userName + '/' + password;
    return this.http.get(url);
  }

  public addDoctor(content){
    const url = '/api/doctor/addDoctor';
    return this.http.post(url,content);
  }

  public addPatient(content){
    const url = '/api/patient/addPatient';
    return this.http.post(url,content);
  }

  public addCenter(content){
    const url = '/api/diagnosiscenter/addCenter';
    return this.http.post(url,content);
  }

  public getAllDocuments(){
    return this.http.get('/api/document/getAllDocuments');
  }



  public uploadFile(file, userId, docType){
    return this.http.post('/api/document/uploadFile/'+file+'/'+userId+'/'+docType, '');
  }
}
