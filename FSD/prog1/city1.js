const prompt = require('prompt-sync')();   
// Prompt the user to enter 5 city names, separated by commas  
let input = prompt("Enter 5 cities separated by commas: ");  
let cities = input.split(',').map(city => city.trim());  
console.log("Initial cities:", cities);  
// Log the total number of cities  
console.log("Total number of cities:", cities.length);   
// Add a new city at the end  
let newCity = prompt("Enter a city to add to the end: ");  
cities.push(newCity);  
console.log("Cities after adding a new one:", cities);   
// Remove the first city  
console.log("Removing the first city:", cities[0]);  
// Log the first city before removal  
cities.shift();  
console.log("Cities after removing the first one:", cities);   
// Find and log the index of a specific city  
let searchCity = prompt("Enter a city to find its index: ");  
let cityIndex = cities.indexOf(searchCity);  
console.log("Index of", searchCity + ":", cityIndex !== -1 ? cityIndex : "City notfound"); 