import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { LoginComponent } from './core/login/login.component';
import { DashboardComponent } from './features/dashboard/dashboard.component';
import { RegistrationComponent } from './core/registration/registration.component';
import { UserProfileComponent } from './core/user-profile/user-profile.component';
import {  AuthGuardService } from './services/auth.guard';

const routes: Routes = [
  { path : '', redirectTo: '/login', pathMatch: 'full'},
  { path: 'login', component: LoginComponent},
  { path: 'register', component: RegistrationComponent},
  { path: 'dashboard', component: DashboardComponent},
  { path: 'userProfile', component: UserProfileComponent},
];


@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
