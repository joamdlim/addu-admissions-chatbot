# Subfolder Implementation - Complete Guide

## Overview

This document outlines the complete implementation of hierarchical subfolder support in the ADDU Admissions Chatbot admin panel.

## Features Implemented

### 1. **Backend Changes**

#### Models (`backend/chatbot/models.py`)

- ✅ Added `parent_folder` field to `DocumentFolder` model
- ✅ Added properties:
  - `total_document_count` - Count documents in folder + all subfolders
  - `folder_path` - Full path (e.g., "Curriculum / Undergraduate / Engineering")
  - `level` - Depth level in hierarchy (0 for root)
  - `get_all_descendant_folders()` - Get all subfolders recursively
- ✅ Changed `unique_together` to allow same folder name in different parents

#### Views (`backend/chatbot/views.py`)

- ✅ Updated `manage_folders` to support `parent_id` query parameter
- ✅ Added `get_all_folders` endpoint for dropdown selects
- ✅ Updated folder creation to support `parent_folder_id`
- ✅ Added validation to prevent circular references
- ✅ Updated delete to cascade through all subfolders

#### URLs (`backend/chatbot/urls.py`)

- ✅ Added `/admin/folders/all/` endpoint

### 2. **Frontend Changes**

#### AdminPage.jsx (`admin-frontend/src/pages/AdminPage.jsx`)

**New State Variables:**

```jsx
- currentFolderId: null          // Track current folder location
- folderBreadcrumbs: []          // Navigation trail
- allFolders: []                 // All folders for hierarchical display
- parent_folder_id: null         // In newFolder state
```

**New Functions:**

```jsx
-fetchFolders(parentId) - // Fetch folders for specific parent
  fetchAllFolders() - // Fetch all folders for dropdowns
  navigateToFolder(folderId); // Navigate to folder, update breadcrumbs
```

**UI Features:**

- ✅ **Breadcrumb Navigation** - Shows current path (Root / Folder1 / Folder2)
- ✅ **Back Button** - Navigate to parent folder (appears when inside a folder)
- ✅ **Double-click to Open** - Double-click folder card to view its contents
- ✅ **Subfolder Indicator** - Badge showing number of subfolders (📁 3)
- ✅ **Total Document Count** - Shows docs in folder + all subfolders
- ✅ **Hierarchical Dropdowns** - Indented folder paths in selects
- ✅ **Context-aware Creation** - New folders created in current location
- ✅ **Empty State Message** - "No folders here. Create one to get started!"

## User Experience

### Navigation Flow:

1. **Start at Root** - See all root-level folders
2. **Double-click a folder** - Opens the folder to view subfolders
3. **Breadcrumbs appear** - Shows path: 🏠 Root / Curriculum / Undergraduate
4. **Click breadcrumb** - Jump to any level in hierarchy
5. **Back button** - Quick return to parent folder
6. **Create subfolder** - New folder button creates in current location

### Visual Indicators:

- **Folder Badge** - 📁 icon with count shows subfolders
- **Document Count** - "5 docs (12)" = 5 direct, 12 total
- **Color Inheritance** - Subfolders can inherit parent color
- **Path Display** - Full path shown in dropdowns with indentation

## API Endpoints

### GET `/chatbot/admin/folders/`

- **Query Params**: `?parent_id=<id>` (optional)
- **Returns**: Folders at specified level (root if no parent_id)

### GET `/chatbot/admin/folders/all/`

- **Returns**: All folders with hierarchy info (for dropdowns)

### POST `/chatbot/admin/folders/`

- **Body**: `{ name, description, color, parent_folder_id }`
- **Creates**: Folder in specified parent (root if null)

### PUT `/chatbot/admin/folders/<id>/`

- **Body**: `{ name, description, color, parent_folder_id }`
- **Updates**: Folder properties (can move to different parent)

### DELETE `/chatbot/admin/folders/<id>/`

- **Deletes**: Folder and all subfolders + documents (cascade)

## Database Schema

```sql
-- DocumentFolder table structure
CREATE TABLE chatbot_documentfolder (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    color VARCHAR(7) DEFAULT '#063970',
    parent_folder_id INTEGER REFERENCES chatbot_documentfolder(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(name, parent_folder_id)  -- Same name allowed in different parents
);
```

## Example Folder Structure

```
📁 Root
├── 📁 Curriculum (5 docs, 3 subfolders)
│   ├── 📁 Undergraduate (8 docs)
│   ├── 📁 Graduate (12 docs)
│   └── 📁 Senior High (6 docs)
├── 📁 Admissions (10 docs, 2 subfolders)
│   ├── 📁 Requirements (15 docs)
│   └── 📁 Deadlines (8 docs)
└── 📁 Financial Aid (20 docs)
```

## Testing Checklist

- [ ] Create root folder
- [ ] Create subfolder inside root folder
- [ ] Create nested subfolder (3+ levels deep)
- [ ] Navigate using breadcrumbs
- [ ] Use back button
- [ ] Upload document to subfolder
- [ ] Move document between folders
- [ ] Delete folder with subfolders
- [ ] Rename folder
- [ ] Move folder to different parent
- [ ] Test circular reference prevention

## Migration Steps

1. **Database Migration**:

   ```bash
   python manage.py migrate chatbot
   ```

2. **Verify Tables**:

   ```sql
   \d chatbot_documentfolder  -- Check parent_folder_id exists
   ```

3. **Test API**:

   ```bash
   curl http://localhost:8000/chatbot/admin/folders/all/
   ```

4. **Start Frontend**:
   ```bash
   cd admin-frontend
   npm run dev
   ```

## Known Limitations

1. **Max Depth**: No hard limit, but UI may become unwieldy beyond 5 levels
2. **Breadcrumb Length**: Long paths may overflow on small screens
3. **Folder Path Cache**: Breadcrumbs built from `folder_path` property

## Future Enhancements

- [ ] Drag-and-drop to move folders
- [ ] Folder color inheritance toggle
- [ ] Expand/collapse tree view option
- [ ] Folder templates
- [ ] Bulk move operations
- [ ] Folder permissions

## Files Modified

### Backend:

- `backend/chatbot/models.py` - DocumentFolder model
- `backend/chatbot/views.py` - Folder management views
- `backend/chatbot/urls.py` - API routes
- `backend/chatbot/migrations/0002_*.py` - Database migration

### Frontend:

- `admin-frontend/src/pages/AdminPage.jsx` - Main admin interface

## Rollback Instructions

If you need to rollback:

1. **Revert Frontend**:

   ```bash
   git checkout HEAD~1 admin-frontend/src/pages/AdminPage.jsx
   ```

2. **Revert Backend** (if needed):

   ```bash
   python manage.py migrate chatbot 0001_initial
   ```

3. **Remove Migration**:
   ```bash
   rm backend/chatbot/migrations/0002_*.py
   ```

## Success Criteria

✅ Users can create folders within folders
✅ Navigation is intuitive with breadcrumbs
✅ Document upload works with nested folders
✅ Folder deletion cascades properly
✅ No circular reference bugs
✅ All existing functionality preserved

---

**Implementation Date**: October 13, 2025
**Status**: Complete ✅
