import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import {ToastModule} from 'primeng/toast';
import { MessageService } from 'primeng/api';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { LoginComponent } from './core/login/login.component';
import { DashboardComponent } from './features/dashboard/dashboard.component';
import { UserProfileComponent } from './core/user-profile/user-profile.component';
import { PostComponent } from './features/post/post.component';
import { CommentsComponent } from './shared/comments/comments.component';
import { TagsComponent } from './shared/tags/tags.component';
import { FeedComponent } from './features/feed/feed.component';
import { HeaderComponent } from './core/header/header.component';
import { NgMultiSelectDropDownModule } from 'ng-multiselect-dropdown';
import { FooterComponent } from './core/footer/footer.component';
import { ActivityComponent } from './features/activity/activity.component';
import { NotificationComponent } from './shared/notification/notification.component';
import { RegistrationComponent } from './core/registration/registration.component';

import {ReactiveFormsModule} from '@angular/forms';

import { MaterialExampleModule } from './material.module';
import { environment } from 'src/environments/environment';
import { NgxAgoraModule } from 'ngx-agora';
import { VideoCallComponent } from './core/video-call/video-call.component'

@NgModule({
  declarations: [
    AppComponent,
    LoginComponent,
    DashboardComponent,
    UserProfileComponent,
    PostComponent,
    CommentsComponent,
    TagsComponent,
    FeedComponent,
    HeaderComponent,
    FooterComponent,
    ActivityComponent,
    NotificationComponent,
    RegistrationComponent,
    VideoCallComponent
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    AppRoutingModule,
    ToastModule,
    NgMultiSelectDropDownModule.forRoot(),
    FormsModule,
    HttpClientModule,
    MaterialExampleModule,
    ReactiveFormsModule,
    NgxAgoraModule.forRoot({ AppID: environment.agora.appId })
  ],
  providers: [MessageService],
  bootstrap: [AppComponent]
})
export class AppModule { }
