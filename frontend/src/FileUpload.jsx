import { useState, useEffect } from "react"
import DetailCard from "./DetailCard"

export default function FileUploader() {

    const [file, setFile] = useState(null)
    const [prediction, setPrediction] = useState(null)

    useEffect(() => {
        if (!file) return

        const formData = new FormData();
        formData.append("file", file)

        fetch(`${import.meta.env.VITE_API_URL}/predict/`, {
            method: "POST",
            body: formData,
        })
                
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`)
                }
                return response.json()
            })

            .then((result) => {
                console.log("Upload successful!")
                setPrediction(result.prediction.Name)
            })

            .catch((error) => {
                console.log("Error occurred", error)
                alert("Upload failed!")
            })

    }, [file])
   
    const handleFileUpload = (e) => {
        const uploadedFile = e.target.files[0];
        if (uploadedFile) setFile(uploadedFile)
    }

    return <>
            <form className = "form-container" action = "upload" method = "post">
                <label className = "label-item" htmlFor = "myfile">Upload your Pokémon by clicking the button below!</label>
                <input type = "file" id = "myfile" className = "upload-item" onChange = {handleFileUpload}/>
            </form>
            {prediction && <DetailCard key = {prediction + Date.now()} pokemon = {prediction}/>}
           </>
}