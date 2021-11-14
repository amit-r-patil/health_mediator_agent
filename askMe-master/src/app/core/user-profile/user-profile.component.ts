import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-user-profile',
  templateUrl: './user-profile.component.html',
  styleUrls: ['./user-profile.component.scss']
})
export class UserProfileComponent implements OnInit {

  constructor() { }
  currentUser: any;
  currentUserQuestion: any = [];
  filteredUserQuestions: any = [];

  userType = localStorage.getItem("userType");
  ngOnInit(): void {
    this.currentUser = JSON.parse(localStorage.getItem('currentUser'));
    const allQuestions = JSON.parse(localStorage.getItem('allQuestions'));
    if (allQuestions) {
      allQuestions.forEach(question => {
        if (question.author === this.currentUser.userName) {
          this.currentUserQuestion.push(question);
          this.filteredUserQuestions.push(question);
        }
      });
    }

  }

  filterData(event: any) {
    this.filteredUserQuestions = [];

    if (event === '') {
      this.filteredUserQuestions = this.currentUserQuestion;
      return;
    }

    this.currentUserQuestion.forEach(quest => {
      const q = quest.tags.filter(tag => tag.id === event);

      if (q.length > 0) {
        this.filteredUserQuestions.push(quest);
      }
    });

  }

}
