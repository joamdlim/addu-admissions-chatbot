import FileUploader from '../components/FileUploader';

const AdminPage = () => {
  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold mb-4">Upload Admission Documents</h1>
      <FileUploader />
    </div>
  );
};

export default AdminPage;
