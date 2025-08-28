import React, { useContext, useState } from 'react'
import "./Sidebar.css"
import {assets} from '../../assets/assets'
import { Context } from '../../context/Context'
import ModelSelector from '../Settings/ModelSelector';
import ThemeSelector from '../Settings/ThemeSelector';
import TemperatureSlider from '../Settings/TemperatureSlider';
import VerbositySlider from '../Settings/VerbositySlider';

// Helper function to determine file type based on extension
const getFileTypeIcon = (fileName) => {
  const extension = fileName.split('.').pop().toLowerCase();
  
  // PDF files
  if (extension === 'pdf') {
    return <i className="fi fi-rr-file-pdf" />;
  }
  
  // Data files
  const dataExtensions = ['csv', 'xlsx', 'xls', 'rdata', 'rds', 'spss', 'sav'];
  if (dataExtensions.includes(extension)) {
    return <i className="fi fi-rr-chart-histogram" />;
  }
  
  // Default file icon
  return <img src={assets.gallery_icon} alt="file icon" />;
};

const Kebab = ({ size=13 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <circle cx="12" cy="5" r="2"/>
    <circle cx="12" cy="12" r="2"/>
    <circle cx="12" cy="19" r="2"/>
  </svg>
);

const ChevronDown = ({ size=16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="m6 9 6 6 6-6" />
  </svg>
);

const ChevronUp = ({ size=16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="m18 15-6-6-6 6" />
  </svg>
);

const RecentEntry = ({ s, isActive, onActivate }) => {
  const [open, setOpen] = useState(false);
  const [isRenaming, setIsRenaming] = useState(false);
  const { sessions, setActiveSessionId } = useContext(Context);
  const live = sessions.find(x => x.id === s.id) || s;
  const [draftTitle, setDraftTitle] = useState(live.title || '');
  const inputRef = React.useRef(null);
  const { showToast } = useContext(Context);
  const toggle = (e) => { e.stopPropagation(); setOpen(v => !v); };
  const close = () => setOpen(false);

  React.useEffect(() => { if (isRenaming && inputRef.current) inputRef.current.focus(); }, [isRenaming]);
  React.useEffect(() => { if (!isRenaming) setDraftTitle(live.title || ''); }, [live.title, isRenaming]);

  // Actions
  const download = (e) => {
    e.stopPropagation();
    const blob = new Blob([JSON.stringify(s, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${(s.title || 'chat').replace(/[^a-z0-9-_]+/gi,'_')}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    close();
  };
  const pin = (e) => { e.stopPropagation(); window.appCtx?.pinSession?.(s.id); close(); };
  const startRename = (e) => { e.stopPropagation(); setDraftTitle(live.title || ''); setIsRenaming(true); close(); };
  const commitRename = () => {
    const val = String(draftTitle || '').trim();
    window.appCtx?.renameSessionWithValue?.(s.id, val);
    setDraftTitle(val);
    setIsRenaming(false);
  };
  const cancelRename = () => { setIsRenaming(false); setDraftTitle(live.title || ''); };
  const remove = (e) => { e.stopPropagation(); window.appCtx?.deleteSession?.(s.id); close(); };

  return (
    <div onClick={onActivate} className={`recent-entry ${isActive ? 'active' : ''}`}>
      <img src={assets.message_icon} alt="" />
      {isRenaming ? (
        <input
          key={`${s.id}-edit`}
          ref={inputRef}
          className="rename-input"
          value={draftTitle}
          onChange={(e)=>setDraftTitle(e.target.value)}
          onClick={(e)=>e.stopPropagation()}
          onKeyDown={(e)=>{
            if (e.key === 'Enter') { e.preventDefault(); commitRename(); }
            if (e.key === 'Escape') { e.preventDefault(); cancelRename(); }
          }}
          onBlur={()=>{
            const liveTitle = String(live.title || '');
            const next = String(draftTitle || '').trim();
            if (next !== liveTitle) commitRename(); else setIsRenaming(false);
          }}
        />
      ) : (
        <p key={`${s.id}-view`} className="title">{(live.title || 'New chat').length > 28 ? (live.title || 'New chat').slice(0, 28) + '...' : (live.title || 'New chat')}</p>
      )}
      {s.pinned && (
        <span className="pin-badge" title="Pinned">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M6 2h12a1 1 0 0 1 1 1v19l-7-3-7 3V3a1 1 0 0 1 1-1z"/></svg>
        </span>
      )}
      <span className="more-btn" onClick={toggle} title="More">
        <Kebab/>
      </span>
      {open && (
        <div className="menu" onClick={(e)=>e.stopPropagation()}>
          <div className="item" onClick={download}>
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 3v10m0 0l-4-4m4 4l4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/><path d="M4 17h16v3H4z"/></svg>
            Download
          </div>
          <div className="item" onClick={pin}>
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 2h12a1 1 0 0 1 1 1v19l-7-3-7 3V3a1 1 0 0 1 1-1z"/></svg>
            Pin
          </div>
          <div className="item" onClick={startRename}>
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25z"/><path d="M20.71 7.04a1.003 1.003 0 0 0 0-1.42l-2.34-2.34a1.003 1.003 0 0 0-1.42 0l-1.83 1.83 3.75 3.75 1.84-1.82z"/></svg>
            Rename
          </div>
          <div className="item" onClick={remove} style={{color:'#b00020'}}>
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 7h12l-1 14H7L6 7zM9 7V5h6v2h5v2H4V7h5z"/></svg>
            Delete
          </div>
        </div>
      )}
    </div>
  );
};

const Sidebar = () => {

    const { sidebarExtended: extended, setSidebarExtended: setExtended, onSent, setRecentPrompt, newChat, sessions, activeSessionId, setActiveSession, openSettings, user, signInWithGoogle, signOut, showAllSessions, setShowAllSessions, isSettingsOpen, uploadedFiles, removeUploadedFile } = useContext(Context)
    const [filesOpen, setFilesOpen] = useState(false);
    const [recentOpen, setRecentOpen] = useState(true);
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [menuOpen, setMenuOpen] = useState(false);

    // Accordion-style handlers - only one section can be open at a time
    const handleRecentToggle = () => {
        const newRecentState = !recentOpen;
        setRecentOpen(newRecentState);
        if (newRecentState) {
            setFilesOpen(false);
            setSettingsOpen(false);
        }
    };

    const handleFilesToggle = () => {
        const newFilesState = !filesOpen;
        setFilesOpen(newFilesState);
        if (newFilesState) {
            setRecentOpen(false);
            setSettingsOpen(false);
        }
    };

    const handleSettingsToggle = () => {
        const newSettingsState = !settingsOpen;
        setSettingsOpen(newSettingsState);
        if (newSettingsState) {
            setRecentOpen(false);
            setFilesOpen(false);
        }
    };

    const loadPrompt = async (prompt) => {
      setRecentPrompt(prompt)
      await onSent(prompt)
    }

  return (
    <div className={`sidebar ${!extended ? 'collapsed' : ''}`}>
      <div className="top">
        <div className="sidebar-header">
          {extended && (
            <div className="esi-logo-sidebar" onClick={()=>newChat()}>
              <img src={assets.network_nodes_logo} alt="Assistant" />
              <span className="esi-text">esi</span>
            </div>
          )}
          <div className={`sidebar-toggle ${!extended ? 'left-aligned' : ''}`}>
            <i
              onClick={() => setExtended(prev => !prev)}
              className="fi fi-rr-sidebar menu"
              title="Toggle sidebar"

            />
          </div>
        </div>
        <div className="sidebar-controls">
          <div className="search-box">
            <i
              className="fa-solid fa-search search-icon"
              onClick={() => !extended && setExtended(true)}
            ></i>
            {extended && (
              <input
                type="text"
                placeholder="Search chats..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="search-input"
              />
            )}
          </div>

          <div className="new-chat" onClick={()=>newChat()} title="New chat">
            <i className="fi fi-rr-plus new-chat-icon" />
            {extended && <p>New Chat</p>}
          </div>
        </div>

        <div className="recent" ref={(el)=>{ /* optional ref */ }}>
          <div className="recent-title" onClick={handleRecentToggle} style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <i className="fi fi-rr-messages"></i>
              {extended && <span className="recent-title-text">Recent</span>}
            </div>
            {extended && (recentOpen ? <ChevronUp /> : <ChevronDown />)}
          </div>
          {extended && recentOpen && (() => {
            const filteredSessions = sessions.filter(s =>
              s.title.toLowerCase().includes(searchQuery.toLowerCase())
            );

            const pinned = filteredSessions.filter(x => x.pinned);
            const others = filteredSessions.filter(x => !x.pinned);

            // Determine how many non-pinned to show when collapsed (fixed default)
            const maxRows = 8;
            const visibleOthers = showAllSessions ? others : others.slice(0, maxRows);

            return (
              <>
                {pinned.length > 0 && (
                  <div className="section">
                    {pinned.map((s) => (
                      <RecentEntry
                        key={s.id}
                        s={s}
                        isActive={s.id === activeSessionId}
                        onActivate={() => setActiveSession(s.id)}
                      />
                    ))}
                  </div>
                )}

                {pinned.length > 0 && others.length > 0 && (
                  <div className="divider"/>
                )}

                {others.length > 0 && (
                  <div className="section">
                    {visibleOthers.map((s) => (
                      <RecentEntry
                        key={s.id}
                        s={s}
                        isActive={s.id === activeSessionId}
                        onActivate={() => setActiveSession(s.id)}
                      />
                    ))}
                    {others.length > visibleOthers.length && (
                      <div className="recent-entry" onClick={() => setShowAllSessions(true)} style={{ justifyContent: 'center', fontWeight: 600 }}>
                        <span>More…</span>
                      </div>
                    )}
                    {showAllSessions && others.length > 0 && (
                      <div className="recent-entry" onClick={() => setShowAllSessions(false)} style={{ justifyContent: 'center', fontWeight: 600 }}>
                        <span>Less</span>
                      </div>
                    )}
                  </div>
                )}
              </>
            );
          })()}
          
          <div className="recent-title" onClick={handleFilesToggle} style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <i className="fi fi-rr-file"></i>
              {extended && <span className="recent-title-text">Uploaded Files</span>}
            </div>
            {extended && (filesOpen ? <ChevronUp /> : <ChevronDown />)}
          </div>
          {extended && filesOpen && (
            <div className="section">
              {uploadedFiles.map((file, index) => (
                <div key={index} className="recent-entry">
                  {getFileTypeIcon(file.name)}
                  <p className="title">{file.name}</p>
                  <span className="more-btn" onClick={() => removeUploadedFile(file.name)} title="Remove file">
                    <img src={assets.x_icon} alt="remove" />
                  </span>
                </div>
              ))}
              {uploadedFiles.length === 0 && <p style={{textAlign: 'center', fontSize: '13px', opacity: 0.7, margin: '10px 0'}}>No files uploaded</p>}
            </div>
          )}

        </div>
        
      </div>
      <div className="bottom">
        {/* Settings section moved here to be above sign-in */}
        <div className="recent-title" onClick={handleSettingsToggle} style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <i className="fa-solid fa-sliders"></i>
            {extended && <span className="recent-title-text">Settings</span>}
          </div>
          {extended && (settingsOpen ? <ChevronDown /> : <ChevronUp />)}
        </div>
        {extended && settingsOpen && (
          <div className="section">
            <div className="setting-row" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
              <ThemeSelector />
            </div>
            <div className="setting-row" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
              <ModelSelector />
            </div>
            <div className="setting-row" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
              <TemperatureSlider />
            </div>
            <div className="setting-row" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
              <VerbositySlider />
            </div>
          </div>
        )}
        {!user ? (
          <div
            className="bottom-item"
            onClick={signInWithGoogle}
            title="Sign in with Google"
          >
            <i className="fa-regular fa-circle-user" aria-hidden="true"></i>
            {extended && <span>Sign in</span>}
          </div>
        ) : (
          <div style={{ position: 'relative', width: '100%' }}>
            <div className="user-avatar-container" onClick={() => setMenuOpen(v => !v)}>
              <img src={user?.user_metadata?.avatar_url || user?.user_metadata?.picture || assets.avatar} 
                   alt={user?.user_metadata?.full_name || user?.email || 'User'} />
              {extended && (
                <span>
                  {user.user_metadata?.full_name || user.email}
                </span>
              )}
            </div>
            {menuOpen && (
              <div 
                role="menu"
                style={{
                  position: 'absolute',
                  left: 0,
                  bottom: '100%',
                  background: 'var(--panel-bg)',
                  color: 'var(--text)',
                  border: '1px solid var(--border)',
                  padding: '6px 0',
                  borderRadius: 8,
                  boxShadow: '0 8px 28px rgba(0,0,0,0.18)',
                  minWidth: 180,
                  zIndex: 30,
                  marginBottom: '5px'
                }}
                onClick={(e) => e.stopPropagation()}
              >
                <div style={{ padding: '10px 12px', borderBottom: '1px solid var(--border)', fontSize: 12 }}>
                  <div style={{ fontWeight: 600, marginBottom: 2 }}>{user.user_metadata?.full_name || 'Account'}</div>
                  <div style={{ opacity: 0.8 }}>{user.email}</div>
                </div>
                <button
                  onClick={() => { setMenuOpen(false); signOut(); }}
                  style={{ 
                    display: 'block', width: '100%', textAlign: 'left', 
                    padding: '10px 12px', background: 'transparent', 
                    fontSize: '12px', color: 'inherit', border: 'none', 
                    cursor: 'pointer', font: 'inherit' }}
                >
                  Sign out
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default Sidebar
