import { useState, useEffect } from "react";
import UploadService from "../services/FileUploadService";

const FileUpload: React.FC = () => {
    const [currentFile, setCurrentFile] = useState<File>();
    const [progress, setProgress] = useState<number>(0);
    const [description, setDescription] = useState<string>("");

    const selectFile = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { files } = event.target;
        const selectedFiles = files as FileList;
        setCurrentFile(selectedFiles?.[0]);
        setProgress(0);
    };

    const upload = () => {
        setProgress(0);
        if (!currentFile) return;
    
        UploadService.upload(currentFile, (event: any) => {
            setProgress(Math.round((100 * event.loaded) / event.total));
        })
        .then((response) => {
            console.log(response);
            setDescription(response.data.description);
        })
        .catch((err) => {
            setProgress(0);
    
            if (err.response && err.response.data && err.response.data.message) {
              setDescription(err.response.data.message);
            } else {
              setDescription("Could not upload the File!");
            }
    
            setCurrentFile(undefined);
        });
    };

    const getMoreDescription = () => {
        
    }

    return (
        <div>
            <div>
                <div>
                <label>
                    <input type="file" accept=".csv" onChange={selectFile} />
                </label>
            </div>
    
            <div>
                <button
                    disabled={!currentFile}
                    onClick={upload}
                >
                    Upload
                </button>
            </div>
        </div>
    
        {currentFile && (
            <div>
                <div>
                    {progress}%
                </div>
            </div>
        )}
    
        {description && (
            <div>
                {description}
            </div>
        )}

        {description && (
            <div>
                <button
                    disabled={!currentFile}
                    onClick={upload}
                >
                    Get More Description
                </button>
        </div>
        )}
    </div>
    );
};

export default FileUpload;
