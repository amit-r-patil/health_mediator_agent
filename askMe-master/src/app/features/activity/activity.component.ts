import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-activity',
  templateUrl: './activity.component.html',
  styleUrls: ['./activity.component.scss']
})
export class ActivityComponent implements OnInit {

  constructor() { }
  allNotifications = [];
  @Input() notifications: any;
  ngOnInit(): void {
    this.getMessagesForCurrentUser();
  }

  getMessagesForCurrentUser(){
    this.allNotifications = JSON.parse(localStorage.getItem('notifications'));
    if (!this.allNotifications){
      this.allNotifications = [];
    }

    this.allNotifications.filter( message => message.to === JSON.parse(localStorage.getItem('currentUser')).name);

  }

}
