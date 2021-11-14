import { Component, OnInit, Output, EventEmitter } from '@angular/core';

@Component({
  selector: 'app-tags',
  templateUrl: './tags.component.html',
  styleUrls: ['./tags.component.scss']
})
export class TagsComponent implements OnInit {

  constructor() { }
  @Output() tag = new EventEmitter();
  currentTag = '';

  ngOnInit(): void {
  }

  selectedTag(tagName: string){
    this.currentTag = tagName;
    this.tag.emit(tagName);
  }

}
