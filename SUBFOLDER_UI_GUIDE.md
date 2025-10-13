# Subfolder UI Guide - Visual Reference

## User Interface Overview

### 1. Root Level View (No folder selected)

```
┌─────────────────────────────────────────────────────────────┐
│  🏠 Document Management                    [+ New Folder]    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────┐ │ ┌─────────────┐ ┌─────────────┐ ┌──────────┐│
│  │  All    │ │ │ Curriculum  │ │  Admissions │ │ Financial││
│  │ Docs    │ │ │ 📝 5 docs   │ │  📝 10 docs │ │   Aid    ││
│  │         │ │ │ 📁 3        │ │  📁 2       │ │ 📝 20    ││
│  └─────────┘ │ └─────────────┘ └─────────────┘ └──────────┘│
│              │  [Edit] [Delete] [Edit] [Delete]             │
│              │  Double-click to open →                       │
└─────────────────────────────────────────────────────────────┘
```

### 2. Inside a Folder (Curriculum selected)

```
┌─────────────────────────────────────────────────────────────┐
│  🏠 Document Management                    [+ New Folder]    │
├─────────────────────────────────────────────────────────────┤
│  🏠 Root / Curriculum                                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐ │ ┌────────┐ │ ┌──────────────┐ ┌───────────┐│
│  │  All    │ │ │  ⬅️    │ │ │Undergraduate│ │  Graduate ││
│  │ Docs    │ │ │  Back  │ │ │ 📝 8 docs   │ │ 📝 12 docs││
│  │         │ │ │        │ │ │             │ │           ││
│  └─────────┘ │ └────────┘ │ └──────────────┘ └───────────┘│
│              │            │  ┌───────────┐                  │
│              │            │  │Senior High│                  │
│              │            │  │ 📝 6 docs │                  │
│              │            │  └───────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### 3. Deeper Nesting (Undergraduate selected)

```
┌─────────────────────────────────────────────────────────────┐
│  🏠 Document Management                    [+ New Folder]    │
├─────────────────────────────────────────────────────────────┤
│  🏠 Root / Curriculum / Undergraduate                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐ │ ┌────────┐ │ ┌──────────────┐ ┌───────────┐│
│  │  All    │ │ │  ⬅️    │ │ │ Engineering │ │  Business ││
│  │ Docs    │ │ │  Back  │ │ │ 📝 15 docs  │ │ 📝 10 docs││
│  │         │ │ │        │ │ │             │ │           ││
│  └─────────┘ │ └────────┘ │ └──────────────┘ └───────────┘│
│              │            │                                 │
│              │            │  No folders here. Create one... │
└─────────────────────────────────────────────────────────────┘
```

## Key UI Elements

### Breadcrumb Navigation

```
┌─────────────────────────────────────────────────────────┐
│  🏠 Root  /  Curriculum  /  Undergraduate                │
│   ↑click     ↑click         ↑current                    │
│  jumps to   jumps to        location                     │
│   root      this folder     (bold)                       │
└─────────────────────────────────────────────────────────┘
```

### Folder Card (with subfolders)

```
┌───────────────────────────┐
│              ✏️ 🗑️         │  ← Edit/Delete buttons
│  🔵 Curriculum            │  ← Color indicator + Name
│  Academic year folders    │  ← Description
│  5 docs (20)      📁 3    │  ← Direct (Total) | Subfolders
└───────────────────────────┘
    ↑                   ↑
  Single click      Double click
  = Filter docs     = Open folder
```

### Upload Document - Folder Dropdown

```
┌─────────────────────────────────────────────────┐
│ Folder *                                        │
│ ┌─────────────────────────────────────────────┐│
│ │ Select folder...                          ▼││
│ │─────────────────────────────────────────────││
│ │ Admissions                                  ││  ← Root level (no indent)
│ │ Curriculum                                  ││  ← Root level
│ │   Undergraduate                             ││  ← 1st level (2 spaces)
│ │     Engineering                             ││  ← 2nd level (4 spaces)
│ │     Business                                ││  ← 2nd level
│ │   Graduate                                  ││  ← 1st level
│ │   Senior High                               ││  ← 1st level
│ │ Financial Aid                               ││  ← Root level
│ └─────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
```

### Create Folder Form (Inside a folder)

```
┌─────────────────────────────────────────────────────────┐
│ Folder Name     │ Description      │ Color              │
│ ┌─────────────┐ │ ┌──────────────┐│ ┌────────┐        │
│ │Engineering  │ │ │UG programs   ││ │ [Blue] │        │
│ └─────────────┘ │ └──────────────┘│ └────────┘        │
│                                                         │
│ ℹ️ Creating subfolder in: Undergraduate                │
│                                                         │
│                         [Create Folder]                 │
└─────────────────────────────────────────────────────────┘
```

## User Actions & Results

### Action 1: Create Root Folder

**Steps:**

1. Click [+ New Folder]
2. Enter name: "Curriculum"
3. Click [Create Folder]

**Result:**

```
✅ Folder 'Curriculum' created successfully
```

### Action 2: Create Subfolder

**Steps:**

1. Double-click "Curriculum" folder
2. Click [+ New Folder]
3. Enter name: "Undergraduate"
4. Click [Create Folder]

**Result:**

```
✅ Folder 'Curriculum / Undergraduate' created successfully
```

Breadcrumb shows: 🏠 Root / Curriculum

### Action 3: Navigate with Breadcrumbs

**Steps:**

1. Currently at: 🏠 Root / Curriculum / Undergraduate
2. Click "Curriculum" in breadcrumb

**Result:**

- View changes to show Curriculum's subfolders
- Breadcrumb updates to: 🏠 Root / Curriculum
- Back button still available

### Action 4: Use Back Button

**Steps:**

1. Currently at: 🏠 Root / Curriculum / Undergraduate
2. Click ⬅️ Back button

**Result:**

- Returns to parent (Curriculum level)
- Breadcrumb updates to: 🏠 Root / Curriculum

### Action 5: Upload to Subfolder

**Steps:**

1. Drag PDF to upload area
2. Select folder: " Undergraduate" (indented in dropdown)
3. Fill metadata
4. Click [Submit Upload]

**Result:**

```
✅ Document uploaded successfully to Curriculum / Undergraduate
```

### Action 6: Delete Folder with Subfolders

**Steps:**

1. Click 🗑️ on "Curriculum" folder
2. Confirm deletion

**Result:**

```
⚠️ Are you sure you want to delete "Curriculum"?
   This will also delete all documents in this folder and subfolders.

   [Cancel]  [Delete]

✅ Folder 'Curriculum' and all subfolders deleted successfully
```

## Visual States

### Empty Folder

```
┌─────────────────────────────────────────────────────────┐
│  🏠 Root / Curriculum / Undergraduate                    │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐ │ ┌────────┐ │                            │
│  │  All    │ │ │  ⬅️    │ │                            │
│  │ Docs    │ │ │  Back  │ │   No folders here.         │
│  └─────────┘ │ └────────┘ │   Create one to get        │
│              │            │   started!                  │
└─────────────────────────────────────────────────────────┘
```

### Folder with Documents but No Subfolders

```
┌───────────────────────────┐
│              ✏️ 🗑️         │
│  🔵 Financial Aid         │
│  Scholarship info         │
│  20 docs                  │  ← No subfolder count shown
└───────────────────────────┘
```

### Folder with Both Documents and Subfolders

```
┌───────────────────────────┐
│              ✏️ 🗑️         │
│  🔵 Curriculum            │
│  Academic programs        │
│  5 docs (26)      📁 3    │  ← Direct (total in tree) | Subfolder count
└───────────────────────────┘
```

## Interaction Legend

| Action               | Effect                            |
| -------------------- | --------------------------------- |
| **Single Click**     | Select folder (filter documents)  |
| **Double Click**     | Open folder (navigate into)       |
| **✏️ Button**        | Edit folder details               |
| **🗑️ Button**        | Delete folder (with confirmation) |
| **Breadcrumb Click** | Jump to that level                |
| **⬅️ Back**          | Go to parent folder               |
| **🏠 Root**          | Return to root level              |

## Color Coding

| Element         | Color | Meaning             |
| --------------- | ----- | ------------------- |
| **Blue Border** | 🔵    | Selected folder     |
| **Gray Border** | ⚪    | Unselected folder   |
| **Blue Badge**  | 📁    | Has subfolders      |
| **Green**       | ✅    | Success message     |
| **Red**         | ❌    | Delete button/Error |
| **Yellow**      | ⚠️    | Warning message     |

## Tips & Tricks

1. **Quick Navigation**: Use breadcrumbs instead of back button for multi-level jumps
2. **Visual Hierarchy**: Indented dropdown shows exact folder structure
3. **Document Count**: "(20)" shows total including subfolders
4. **Empty Folders**: Can exist - useful for future organization
5. **Rename Safety**: Prevents duplicate names within same parent
6. **Delete Warning**: Always confirms before deleting folders with content

---

**Note**: All folder operations are immediate - no "Save" button needed!
