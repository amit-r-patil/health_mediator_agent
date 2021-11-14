import { Component, OnInit } from '@angular/core';
import { CoreService } from '../../services/core.service';
import { Router } from '@angular/router';
import { MessageService } from 'primeng/api';


@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss']
})
export class LoginComponent implements OnInit {

  constructor(private coreService: CoreService, private route: Router, private messageService: MessageService) { }

  userName: string;
  password: string;
  errorMessage: string;
  allUsers: Array<any> = [];
  userType: string;

  ngOnInit(): void {
    this.errorMessage = '';
    this.getUsers();
  }

  getUsers() {
    this.coreService.getUser()
      .subscribe(response => {
        console.log(response);
        this.allUsers = response;
      }, error => {
        console.log(error);
      });
  }



  loginToApp() {

    this.errorMessage = '';

    if (!this.userType) {
      this.errorMessage = "Select your User Type";
      return false;
    }


    /** For doc */

    if (this.userType == "doctor") {
      this.coreService.authenticateDoctor(this.userName, this.password)
        .subscribe(res => {
          if (res) {
            localStorage.setItem("userType", "doctor");
            localStorage.setItem("userName", this.userName);
            localStorage.setItem("userInfo", JSON.stringify(res));
            this.route.navigateByUrl('/dashboard');
          } else {
            this.messageService.add({
              severity: 'error',
              summary: 'Invalid Credentials',
              detail: 'Please enter correct details'
            })
          }
        }, error => {
          this.messageService.add({
            severity: 'error',
            summary: 'Ohhoo',
            detail: 'We are experiencing some problems, please try again'
          })
        })
    }


    /** for user */


    if (this.userType == "user") {
      this.coreService.authenticateUser(this.userName, this.password)
        .subscribe(res => {
          if (res) {
            localStorage.setItem("userType", "user");
            localStorage.setItem("userName", this.userName);
            localStorage.setItem("userInfo", JSON.stringify(res));
            this.route.navigateByUrl('/dashboard');
          } else {
            this.messageService.add({
              severity: 'error',
              summary: 'Invalid Credentials',
              detail: 'Please enter correct details'
            })
          }
        }, error => {
          this.messageService.add({
            severity: 'error',
            summary: 'Ohhoo',
            detail: 'We are experiencing some problems, please try again'
          })
        })
    }


    /** for center */

    if (this.userType == "center") {
      this.coreService.authenticateCenter(this.userName, this.password)
        .subscribe(res => {
          if (res) {
            debugger
            localStorage.setItem("userType", "center");
            localStorage.setItem("userName", this.userName);
            localStorage.setItem("userInfo", JSON.stringify(res));
            this.route.navigateByUrl('/dashboard');
          } else {
            this.messageService.add({
              severity: 'error',
              summary: 'Invalid Credentials',
              detail: 'Please enter correct details'
            })
          }
        }, error => {
          this.messageService.add({
            severity: 'error',
            summary: 'Ohhoo',
            detail: 'We are experiencing some problems, please try again'
          })
        })
    }


    const result = this.validUser(this.userName, this.password, this.userType);
    if (result) {
      localStorage.setItem('currentUser', JSON.stringify(result));
      this.route.navigateByUrl('/dashboard');

    } else {
      //this.errorMessage = 'Invalid Details, please try with correct details';
    }
  }

  validUser = (username: string, password: string, userType: string) => {

    if (this.allUsers.length === 0) { return false; }
    let loggedInUser = '';
    this.allUsers.forEach(user => {
      if ((user.userName === username && user.password === password && user.type === userType)
        || (user.userEmail === username && user.password === password && user.type === userType)) {
        loggedInUser = user;
        localStorage.setItem("userType", userType);
        return;
      }
    });


    return loggedInUser;
  }

  registerToApp() {
    this.route.navigateByUrl('/register')
  }
}
