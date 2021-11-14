import { Injectable } from '@angular/core';
import { Router, CanActivate } from '@angular/router';

@Injectable({
    providedIn: 'root'
})
export class AuthGuardService implements CanActivate {
  constructor(public router: Router) {}
  canActivate(): boolean {
    const user = JSON.parse(localStorage.getItem('currentUser'));
    if (!user) {
      this.router.navigate(['login']);
      return false;
    }
    return true;
  }
}
