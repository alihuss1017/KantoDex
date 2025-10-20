const API_URL = import.meta.env.VITE_API_URL

export default function FileUploader() {

    const handleFileUpload = async(e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file)

        try {
            const response = await fetch(`${API_URL}/predict/`,
                {
                    method: "POST",
                    body: formData,
                });
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status})`)
            }
            const result = await response.json();
            console.log("upload succesful")
            alert(`Prediction: ${result.prediction.Name}`)
        }

        catch (error) {
            console.log(`Error occured: ${error}`)
            alert('Upload failed.')
        }
    }

    return <form className = "form-container" action = "upload" method = "post">
           <label className = "label-item" htmlFor = "myfile">Upload your Pokémon by clicking the button below!</label>
           <input type = "file" id = "myfile" className = "upload-item" onChange = {handleFileUpload}/>
           </form>
}