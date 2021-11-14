import { Component, OnInit } from '@angular/core';
import { FormControl } from '@angular/forms';
import { Router } from '@angular/router';
import { CoreService } from '../../services/core.service';
import { IDropdownSettings } from 'ng-multiselect-dropdown';
import { MessageService } from 'primeng/api';


@Component({
  selector: 'app-registration',
  templateUrl: './registration.component.html',
  styleUrls: ['./registration.component.scss']
})
export class RegistrationComponent implements OnInit {

  constructor(private route: Router, private coreService: CoreService, private messageService: MessageService) { }
  userType: string
  userName: string;
  name: string;
  password: string;
  userIdType: any;
  errorMessage: string;
  allUsers: Array<any> = [];
  userIdNumber: any;
  formType: string = '';
  address: any;
  contactNumber: any;
  docHospital: any;
  emailId: any;
  centerServices = [];


  userGender: any;
  userDOB: any;
  userHeight: any;
  userWeight: any;

  docRegNumber: any;
  docQualification: any;
  docSpecialization: any;

  centerRegNumber: any;
  daysAvailable;


  dropdownSettings: IDropdownSettings = {
    singleSelection: true,
    idField: 'id',
    textField: 'value',
    selectAllText: 'Select All',
    unSelectAllText: 'UnSelect All',
    allowSearchFilter: false
  };


  dropdownSettingsCenter: IDropdownSettings = {
    singleSelection: false,
    idField: 'id',
    textField: 'value',
    selectAllText: 'Select All',
    unSelectAllText: 'UnSelect All',
    allowSearchFilter: true
  };

  dropdownSettingsDays: IDropdownSettings = {
    singleSelection: false,
    idField: 'id',
    textField: 'value',
    selectAllText: 'Select All',
    unSelectAllText: 'UnSelect All',
    allowSearchFilter: true
  };


  listDays = [
    { id: 'monday', value: 'Monday' },
    { id: 'tuesday', value: 'Tuesday' },
    { id: 'wednesday', value: 'Wednesday' },
    { id: 'thursday', value: 'Thursday' },
    { id: 'friday', value: 'Friday' },
    { id: 'saturday', value: 'Saturday' },
    { id: 'sunday', value: 'Sunday' }

  ]
  dropdownList = [
    { id: 'pan', value: 'PAN card' },
    { id: 'adhar', value: 'Adhar card' },
    { id: 'passport', value: 'Passport' },
    { id: 'drivingLicense', value: 'Driving License' },
    { id: 'votingcard', value: 'Voting Card' },
  ];

  dropdownListCenter = [
    { id: 'xray', value: 'XRay' },
    { id: 'bloodtest', value: 'Blood test' },
    { id: 'mri', value: 'MRI' },
    { id: 'sonography', value: 'Sonography' },
    { id: 'vitamintest', value: 'Vitamin Test' }
  ]

  ngOnInit(): void {
  }

  updateForm() {
    this.formType = this.userType;
  }


  onItemSelect(item: any) {

  }
  onSelectAll(items: any) {

  }

  onItemDeSelect(item: any) {

  }

  onDeSelectAll(item: any) {

  }

  onItemSelectDays(item: any) {
  }
  onSelectAllDays(items: any) {

  }

  onItemDeSelectDays(item: any) {

  }

  onDeSelectAllDays(item: any) {

  }

  loginToApp() {
    this.route.navigateByUrl('/login')
  }


  register() {
    this.errorMessage = '';
    if (!this.userType) {
      this.errorMessage = 'Please select user type'
      return false;
    }


    if (this.userType == "center") {
      if (this.name && this.userName && this.password && this.address && this.contactNumber
        && this.emailId && this.centerServices && this.daysAvailable && this.centerRegNumber) {
        let regObj = {
          "name": this.name,
          "registrationNumber": this.centerRegNumber,
          "username": this.userName,
          "password": this.password,
          "address": this.address,
          "lat": 0,
          "lon": 0,
          "contactNumber": this.contactNumber,
          "email": this.emailId,
          "daysAvailable": "Monday,Tuesday",
          "timeAvailable": "09.00AM-5:00PM",
          "services": "Blood,XRAY"
        }

        this.coreService.addCenter(regObj)
          .subscribe(response => {
            if (response) {
              this.messageService.add({
                severity: 'success',
                summary: 'Success',
                detail: 'You can now login!'
              })

              this.clearAll();
            }
          }, err => {
            this.messageService.add({
              severity: 'error',
              summary: 'Error',
              detail: 'Please try again'
            })
          })

      } else {
        this.errorMessage = "Please fill all the data"
      }
    }


    if (this.userType == " user") {
      if (this.name && this.userName && this.address && this.password && this.userIdNumber
        && this.userIdType && this.contactNumber && this.userGender && this.userHeight && this.userWeight && this.userDOB) {
        let regObj = {
          "name": this.name,
          "address": this.address,
          "idProofType": this.userIdType,
          "idProofNumber": this.userIdNumber,
          "contactNumber": this.contactNumber,
          "email": this.emailId,
          "username": this.userName,
          "password": this.password,
          "gender": this.userGender,
          "dateOfBirth": this.userDOB,
          "height": this.userHeight,
          "weight": this.userWeight
        }

        this.coreService.addPatient(regObj)
          .subscribe(response => {
            if (response) {
              this.messageService.add({
                severity: 'success',
                summary: 'Success',
                detail: 'You can now login!'
              })
              this.clearAll();
            }
          }, err => {
            this.messageService.add({
              severity: 'error',
              summary: 'Error',
              detail: 'Please try again'
            })
          })

      } else {
        this.errorMessage = "Please fill all the data"
      }
    }

    if (this.userType == "doctor") {
      if (this.name && this.userName && this.password && this.address && this.contactNumber
        && this.emailId && this.daysAvailable && this.docSpecialization && this.docHospital) {

        let regObj = {
          "docName": this.name,
          "registrationNumber": this.docRegNumber,
          "qualification": this.docQualification,
          "specialization": this.docSpecialization,
          "username": this.userName,
          "address": this.address,
          "lat": 19.1267,
          "lon": 72.8754,
          "contactNumber": this.contactNumber,
          "email": this.emailId,
          "daysAvailable": "Monday,Tuesday",
          "timeAvailable": "09.00AM-5:00PM",
          "hospitalName": this.docHospital,
          "password": this.password
        }

        this.coreService.addDoctor(regObj)
          .subscribe(response => {
            if (response) {
              this.messageService.add({
                severity: 'success',
                summary: 'Success',
                detail: 'You can now login!'
              })

              this.clearAll();
            }
          }, err => {
            this.messageService.add({
              severity: 'error',
              summary: 'Error',
              detail: 'Please try again'
            })
          })

      } else {
        this.errorMessage = "Please fill all the data"
      }
    }
  }

  clearAll(){
    this.name = '';
    this.userName = '';
    this.password = '';
    this.userDOB = '';
    this.userGender = '';
    this.userHeight = '';
    this.daysAvailable = '';
    this.address = '';
    this.contactNumber = '';
    this.emailId = '';
    this.docSpecialization = '';
    this.docHospital = '';
    this.userIdNumber = '';
    this.userIdType = '';
    this.centerRegNumber = '';
    this.docRegNumber = '';
  }

}
