import { elementEventFullName } from '@angular/compiler/src/view_compiler/view_compiler';
import { Component, OnInit, Output, EventEmitter } from '@angular/core';
import { IDropdownSettings } from 'ng-multiselect-dropdown';
import { MessageService } from 'primeng/api';

import { CoreService } from '../../services/core.service';

interface QuestionInterface {
  'id': number;
  'content': string;
  'date': any;
  'author': string;
  'comment'?: any;
  'tags'?: Array<any>;
  'answers'?: Array<any>;
  'file'?:any;
  'currentAnswered'?: boolean;
}

@Component({
  selector: 'app-post',
  templateUrl: './post.component.html',
  styleUrls: ['./post.component.scss'],
  providers: [MessageService]
})
export class PostComponent implements OnInit {

  dropdownList = [];
  selectedTags = [];
  question: string;
  user: any;
  userType: any;
  questionUser: any;
  dropdownSettings: IDropdownSettings;
  dropdownSettingsDoctors: IDropdownSettings;
  dropdownListDoc: any;
  dropdownListPatient: any;
  startCall: any = false;
  patientEmail: any;
  allPatients = [];
  fileUploaded: any;
  patientName: any;
  reportName: any;
  fileName: any;
  @Output() questionContent = new EventEmitter();

  constructor(private messageService: MessageService, private coreService: CoreService) { }
  ngOnInit() {

    this.user = JSON.parse(localStorage.getItem('currentUser'));
    this.userType = localStorage.getItem("userType");

    if (this.userType === 'user') {
      this.getAllDoctors();
    }


    if (this.userType === 'center' || this.userType == 'doctor') {
      this.getAllPatients();
    }

    this.dropdownList = [
      { id: 'xray', value: 'XRay' },
      { id: 'bloodtest', value: 'Blood test' },
      { id: 'mri', value: 'MRI' },
      { id: 'sonography', value: 'Sonography' },
      { id: 'vitamintest', value: 'Vitamin Test' },
    ];


    this.dropdownSettingsDoctors = {
      singleSelection: true,
      idField: 'id',
      textField: 'value',
      selectAllText: 'Select All',
      unSelectAllText: 'UnSelect All',
      allowSearchFilter: true
    };
    this.dropdownSettings = {
      singleSelection: false,
      idField: 'id',
      textField: 'value',
      selectAllText: 'Select All',
      unSelectAllText: 'UnSelect All',
      allowSearchFilter: true
    };
  }

  getAllDoctors() {
    let dropdownListDoc1 = [
      { id: 'xry', value: 'Dr. XYC, MBBS' },
      { id: 'abc', value: 'DR. ABC, Derma' },
      { id: 'def', value: 'DR. DEF, Physician' },
      { id: 'hij', value: 'DR HIJ, Ortho' },
      { id: 'xyz', value: 'DR. XYZ, wellness' },
    ]

    this.coreService.getAllDoctors()
      .subscribe(response => {
        let responseDocs = [];
        response.forEach(el => {
          let docObj = { id: el.id, value: el.username + ", " + el.qualification };
          responseDocs.push(docObj);

        })

        this.dropdownListDoc = responseDocs;
      }, error => {
        this.dropdownListDoc = dropdownListDoc1;
      })
  }
  getAllPatients() {
    this.coreService.getAllPatients()
      .subscribe(response => {
        let responsePat = [];

        response.forEach(el => {
          let obj = { id: el.id, value: el.name };
          responsePat.push(obj);
          this.allPatients.push(el);
        })

        this.dropdownListPatient = responsePat;
      }, error => {
        console.log(error)
      })
  }

  onItemSelect(item: any) {
    this.selectedTags.push(item);
  }

  onItemSelectPatient(item: any) {
    this.allPatients.forEach(el => {
      if (el.id == item.id) {
        this.patientEmail = el.email
      }
    })
  }

  onItemSelect1(item: any) {

  }

  onItemDeSelect1(item: any) { }
  onSelectAll(items: any) {
    this.selectedTags = items;
  }

  onItemDeSelect(item: any) {
    console.log(item);
    this.selectedTags = this.selectedTags.filter(tag => {
      return tag.id !== item.id;
    });

    console.log(this.selectedTags);
  }

  onDeSelectAll(item: any) {
    this.selectedTags = [];
  }

  addReport(files) {
    debugger
    this.fileUploaded = false;
    this.fileName = '';
    if (files) {
      this.fileName = files[0].name;
      this.fileUploaded = true;
      let fileList: FileList = files;
      if (fileList.length > 0) {
        let file: File = fileList[0];
        let formData: FormData = new FormData();
        formData.append('file', file, file.name);
        
        let user = JSON.parse(localStorage.getItem('userInfo'));


        this.coreService.uploadFile(formData, 'DC1','jpeg')
          .subscribe(res =>{
            debugger
          })
      }
    }
  }
  addQuestion() {
    if (this.userType == 'center' && (!this.patientEmail || !this.fileUploaded)) {
      this.messageService.add({
        severity: 'error',
        summary: 'Details Missing',
        detail: 'Please enter all details.'
      });

      return
    }
    const question: QuestionInterface = {
      id: new Date().getMilliseconds(),
      content: this.question,
      date: new Date().toDateString(),
      answers: [],
      tags: this.selectedTags,
      author: localStorage.getItem("userName"),
      comment: [],
      currentAnswered: false,
      file: this.fileName
    };
    this.messageService.add({
      severity: 'success',
      summary: 'Report Submitted',
      detail: 'Report submitted!!'
    });



    this.questionContent.emit(question);
    this.clearAll();
  }

  clearAll() {
    this.question = '';
    this.patientEmail = '';
    this.patientName = '';
    this.fileUploaded = '';
    this.reportName = ''
  }

  startVideo() {
    this.startCall = true;
  }

}
