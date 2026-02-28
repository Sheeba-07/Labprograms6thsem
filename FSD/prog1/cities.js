let input = prompt("Enter cities separated by commas:"); 
let cities = input.split(',').map(city => city.trim()); 
let output = document.getElementById("output"); 
let newCity = prompt("Enter a city to add:"); 
cities.push(newCity); 
let removedCity = cities.shift(); 
let searchCity = prompt("Enter city to find:"); 
let index = cities.indexOf(searchCity); 
output.innerHTML = ` 
<p><b>Cities:</b> ${cities.join(", ")}</p> 
<p><b>Total cities:</b> ${cities.length}</p> 
<p><b>Removed city:</b> ${removedCity}</p> 
<p><b>Index of ${searchCity}:</b>  
${index !== -1 ? index : "City not found"}</p>`;