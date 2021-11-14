import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent implements OnInit {

  constructor() { }

  allQuestions = [];
  subComment: string;
  currentUser: any;
  questionsToAnswer = [];
  filteredQuestionsToAnswer = [];
  notifications = [];
  userType = localStorage.getItem("userType");

  ngOnInit(): void {

    this.currentUser = JSON.parse(localStorage.getItem('currentUser'));
    this.allQuestions = JSON.parse(localStorage.getItem('allQuestions'));
    if (!this.allQuestions){ this.allQuestions = []; }
    else{ this.getQuestionToAnswer(this.allQuestions); }
  }

  questionAdded(event: any){
    this.allQuestions.push(event);
    this.getQuestionToAnswer(this.allQuestions);

    localStorage.setItem('allQuestions', JSON.stringify(this.allQuestions));
  }

  getQuestionToAnswer(question: Array<any>){
    if (this.allQuestions){
      this.allQuestions.forEach( quest => {
        if (quest.author !== localStorage.getItem("userName")){
          this.questionsToAnswer.push(quest);
          this.filteredQuestionsToAnswer.push(quest);
        }
      });
    }
  }

  fetchNotifications(event){
    this.notifications = event;
  }

  filterTag(event: any) {
    this.filteredQuestionsToAnswer = [];

    if (event === '') {
      this.filteredQuestionsToAnswer = this.questionsToAnswer;
      return;
    }

    this.questionsToAnswer.forEach(question => {
      const q = question.tags.filter(tag => tag.id === event);
      if (q.length > 0) {
        this.filteredQuestionsToAnswer.push(question);
      }
    });

  }
}
