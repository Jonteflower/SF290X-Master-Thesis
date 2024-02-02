const fs = require('fs');
const Papa = require('papaparse');
const csvFilePath = 'paper_values.csv'

// Function to read csv which returns a promise so you can do async / await.
const readCSV = async (filePath) => {
    const csvFile = fs.readFileSync(filePath)
    const csvData = csvFile.toString()

    return new Promise(resolve => {
        Papa.parse(csvData, {
            header: true,
            complete: results => {
                //console.log('Complete', results.data.length, 'records.'); 
                resolve(results.data);
            }
        });
    });
};

function isIncreasing(arr) {
    return (
        arr.every((v, i) => i === 0 || v >= arr[i - 1])
    );
};

function range(start, end, step){
    let output = [];
    if (typeof end === 'undefined') {
      end = start;
      start = 0;
    }
    for (let i = start; i < end; i += step) {
      output.push(Math.round(i*100)/100);
    }
    return output;
  };

function findPointOfIncrease(data, T_val, sigma_val) {
    const filteredData = data.filter((obj) => Number(obj.sigma) == Number(sigma_val) && Number(obj.T) == Number(T_val))
    const sortedData = filteredData.sort((a, b) => Number(b.H_percent) - Number(a.H_percent))
    //console.log(sortedData.map((obj) => ({ H_percent: Number(Math.round(obj.H_percent * 100) / 100), error_percent: Number(obj.error_percent), })))

    if(filteredData.length == 0){
        return 0
    }
    for (let i = 0; i < sortedData.length; i++) {
        // Split array at index i to get the remaining numbers 
        const splitArray = sortedData.slice(i)

        // Map only the error_percentage into a new Number type array 
        const mappedNumbers = splitArray.map((obj) => (Number(obj.error_percent)))

        if (isIncreasing(mappedNumbers)) {
            return Number(sortedData[i+1].H_percent).toFixed(2)
        }
    }
}

async function test() {
    const parsedData = await readCSV(csvFilePath);
    const filteredData = parsedData.filter((obj) => Number(obj.K) == 300)
    const T_range = range(0.2, 5, 0.1)
    const sigma_range = range(0.2, 0.6,0.05)
    
    const arr = []
    for(const T of T_range){
        for(const sigma of sigma_range){
            const H_value = findPointOfIncrease(filteredData, T, sigma)
            if(Number(H_value) > 0){
                arr.push({h_percent:H_value, product: Number(sigma)*Number(Math.sqrt(T))})
            }   
        }
    }
    console.log(arr.sort((a,b)=>(a.h_percent-b.h_percent)))
}

test()