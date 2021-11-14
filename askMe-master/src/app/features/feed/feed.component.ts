import { Component, OnInit, Input, Output, EventEmitter, OnChanges, SimpleChanges } from '@angular/core';

interface AnswerInterface{
  'id': number;
  'content': string;
  'date': any;
  'author': string;
  'subComment'?: any;
  'tags'?: Array<any>;
}

interface Notification{
  'from': string;
  'to': any;
  'message': any;
}


@Component({
  selector: 'app-feed',
  templateUrl: './feed.component.html',
  styleUrls: ['./feed.component.scss']
})
export class FeedComponent implements OnInit, OnChanges {

  constructor() { }
  @Input() feed: any;
  @Output() notification = new EventEmitter();
  answer = '';
  allNotifications = [];
  file: any;
  ngOnInit(): void {
    console.log(this.feed);
  }

  ngOnChanges(change: SimpleChanges){
debugger
    if (change.feed){
      this.feed = change.feed.currentValue;
    }
  }

  putAnswer(item: any){
    this.feed.forEach(element => {
      element.currentAnswered = false;
    });
    item.currentAnswered = true;
  }

  cancelAnswer(item: any){
    this.feed.forEach(element => {
      element.currentAnswered = false;
    });
  }

  submitAnswer(item: any){
    const currentUser = localStorage.getItem("userName");
    const answer: AnswerInterface = {
      id : Math.random(),
      author : currentUser,
      content: this.answer,
      date: new Date(),
      subComment: ''
    };

    const notification: Notification = {
      to : item.author,
      from : currentUser,
      message : currentUser + ' commented  \'' +  item.content.substring(0, 50) + '...\''
    };

    this.setNotification(notification);

    item.answers.push(answer);
    item.currentAnswered = false;
    localStorage.setItem('allQuestions', JSON.stringify(this.feed));
  }

  setNotification(notification: Notification){
    this.allNotifications = JSON.parse(localStorage.getItem('notifications'));
    if (!this.allNotifications){ this.allNotifications = []; }
    this.allNotifications.push(notification);
    localStorage.setItem('notifications', JSON.stringify(this.allNotifications));
    this.notification.emit(this.allNotifications);
  }
}
