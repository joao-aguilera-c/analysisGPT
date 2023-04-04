import React, { FC, useState } from 'react';
import FileUploadSection from "./components/FileUploadSection";

const App: FC = () => {

    const [description, setDescription] = useState("");

    return (
        <div className="App">
            <FileUploadSection />
        </div>
    );
}

export default App;
