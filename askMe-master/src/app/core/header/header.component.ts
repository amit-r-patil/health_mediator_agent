import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { Route } from '@angular/compiler/src/core';

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.scss']
})
export class HeaderComponent implements OnInit {

  constructor( private router: Router) { }
  currentUser: any;
  ngOnInit(): void {
    this.currentUser = localStorage.getItem('userName');
  }

  appLogout(){
    localStorage.removeItem('userName');
    localStorage.removeItem('userType');
    this.router.navigateByUrl('/login');
  }

  navigateTo(url : string){
    this.router.navigateByUrl(url);
  }


}
