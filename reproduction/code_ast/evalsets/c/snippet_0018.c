/*
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/*
 * uls_context.c -- the input manipulating procedures of ULS --
 *     written by <NAME> <<EMAIL>>, April 2011.
 *
 *  This file is part of ULS, Unified Lexical Scheme.
 */
#define __ULS_CONTEXT__
#include "uls/uls_context.h"
#include "uls/uls_misc.h"
#include "uls/uls_log.h"

ULS_DECL_STATIC int
__xcontext_binfd_filler(uls_xcontext_ptr_t xctx)
{
	uls_input_ptr_t inp = xctx->context->input;
	int  n, rc;

	uls_regulate_rawbuf(inp);

	if ((rc = inp->rawbuf.siz - inp->rawbuf_bytes) <= 0) {
		err_log("%s: InternalError: No space to fill in buffer!", __func__);
		return 0;
	}

	if ((n=uls_input_readn(uls_ptr(inp->isource), inp->rawbuf.buf, inp->rawbuf_bytes, inp->rawbuf.siz)) <= 0) {
		if (n==0) {
			xctx->context->flags |= ULS_CTX_FL_EOF;
		} else {
			return -1;
		}
	}

	inp->rawbuf_bytes += n;

	return inp->rawbuf_bytes;
}

ULS_DECL_STATIC void
add_bin_packet_to_zbuf(int tok_id, int txtlen, const char* txtptr, csz_str_ptr_t ss_dst)
{
	uls_int32 hdrbuf[2];
	char *pktptr;
	int k0;

	k0 = csz_length(ss_dst);
	csz_modify(ss_dst, k0, nilptr, ULS_RDPKT_SIZE);

	hdrbuf[0] = tok_id;
	hdrbuf[1] = txtlen;
	pktptr = (char *) uls_memcopy((csz_data_ptr(ss_dst) + k0), hdrbuf, sizeof(hdrbuf));
	uls_memcopy(pktptr, &txtptr, sizeof(const char *));
}

ULS_DECL_STATIC void
insert_txt_record_into_stream(
	int tokid, int tokid_TMPL, const char *txtptr, int txtlen,
	uls_context_ptr_t ctx, csz_str_ptr_t  ss_dst)
{
	uls_tmplvar_ptr_t tmpl;
	uls_rd_packet_ptr_t pkt;
	int i;

	if (tokid == tokid_TMPL) {
		if ((tmpl = uls_search_tmpls(uls_ptr(ctx->tmpls_pool->tmplvars), txtptr)) == nilptr) {
			add_bin_packet_to_zbuf(tokid, txtlen, txtptr, ss_dst);
		} else {
			for (i=0; i<tmpl->n_pkt_ary; i++) {
				pkt = uls_array_get_slot_type10(uls_ptr(ctx->tmpls_pool->pkt_ary), tmpl->i0_pkt_ary + i);
				add_bin_packet_to_zbuf(pkt->tok_id, pkt->len_tokstr, pkt->tokstr, ss_dst);
			}
		}
	} else {
		add_bin_packet_to_zbuf(tokid, txtlen, txtptr, ss_dst);
	}
}

ULS_DECL_STATIC int
get_txthdr_1(uls_outparam_ptr_t parms)
{
	char *lptr = parms->line;
	const char *lptr_end = parms->lptr_end;
	const char *txtptr;
	int i, txtlen;
	uls_outparam_t parms2;

	for (i = 0; lptr[i]==' '; i++)
		/* NOTHING */;

	parms2.n = i;
	parms->n = splitint(lptr, uls_ptr(parms2));
	parms->len = txtlen = splitint(lptr, uls_ptr(parms2));

	i = parms2.n + 1; // next to ' '
	txtptr = lptr + i;

	lptr += i + txtlen;
	if (lptr >= lptr_end) return -1;

	*lptr++ = '\0';
	parms->line = lptr;
	parms->lptr = txtptr;

	return 0;
}

ULS_DECL_STATIC int
get_txthdr_2(uls_outparam_ptr_t parms)
{
	char *lptr = parms->line, *txtptr, *wrd;
	const char *lptr_end = parms->lptr_end;
	uls_wrd_t wrdx;
	int txtlen;

	wrdx.lptr = lptr;

	if (*(wrd = _uls_splitstr(uls_ptr(wrdx))) == '\0') {
		return -1;
	}
	parms->n = uls_atoi(wrd);

	if (*(wrd = _uls_splitstr(uls_ptr(wrdx))) == '\0') {
		return -1;
	}
	parms->len = txtlen = uls_atoi(wrd);
	txtptr = lptr;

	if (lptr_end < txtptr + txtlen) {
		return -1;
	}

	lptr = txtptr + txtlen;

	*lptr++ = '\0';
	parms->line = lptr;
	parms->lptr = txtptr;

	return 0;
}

ULS_DECL_STATIC int
__xcontext_txtfd_filler(uls_xcontext_ptr_t xctx, uls_outparam_ptr_t parms)
{
	uls_input_ptr_t   inp = xctx->context->input;
	csz_str_ptr_t ss_dst = uls_ptr(xctx->context->zbuf1);
	uls_context_ptr_t ctx = xctx->context;
	int   tok_id, n, rc, rc2, txtlen, n_recs=0;
	char *lptr0, *lptr, *lptr_end;
	const char *txtptr;
	uls_outparam_t parms2;

	//         inp->rawbuf_bytes < inp->rawbuf_siz
	uls_regulate_rawbuf(inp);

again_1:
	n = inp->rawbuf.siz - inp->rawbuf_bytes;
	lptr = inp->rawbuf.buf + inp->rawbuf_bytes;

	if ((rc=uls_input_readn(uls_ptr(inp->isource), inp->rawbuf.buf, inp->rawbuf_bytes, inp->rawbuf.siz)) < 0) {
		if (inp->rawbuf_bytes > 0)
			err_log("%s: redundant %d-bytes exist!", __func__, inp->rawbuf_bytes);
		return -1;

	} else if (rc == 0) {
		if (inp->rawbuf_bytes == 0) {
			parms->n = n_recs;
			return 0;
		}

		lptr = inp->rawbuf.buf;
		lptr_end = lptr + inp->rawbuf_bytes;
		*lptr_end = '\0';

		for ( ; lptr < lptr_end; ) {
			// No worry about the end of record
			parms2.line = lptr;
			parms2.lptr_end = lptr_end;

			rc2 = get_txthdr_2(uls_ptr(parms2));

			lptr = parms2.line;
			tok_id = parms2.n;
			txtlen = parms2.len;
			txtptr = parms2.lptr;

			if (rc2 < 0) {
				err_log("%s: record truncated!", __func__);
				return -1;
			}

			insert_txt_record_into_stream(tok_id, xctx->toknum_TMPL, txtptr, txtlen,
				ctx, ss_dst);

			++n_recs;
		}

		inp->rawbuf_ptr = inp->rawbuf.buf;
		inp->rawbuf_bytes = 0;

		parms->n = n_recs;
		return 1;
	}

	inp->rawbuf_ptr = lptr = inp->rawbuf.buf;
	inp->rawbuf_bytes += rc;

	lptr_end = lptr + inp->rawbuf_bytes;

	for ( ; (n = (int) (lptr_end - lptr)) >= ULS_TXT_RECHDR_SZ; ) {
		lptr0 = lptr;

		parms2.line = lptr;
		parms2.lptr_end = lptr_end;

		rc2 = get_txthdr_1(uls_ptr(parms2));

		lptr = parms2.line;
		tok_id = parms2.n;
		txtlen = parms2.len;
		txtptr = parms2.lptr;

		if (rc2 < 0) {
			if (n_recs > 0) break;

			// 'line' must include at least one record.
			inp->rawbuf_ptr = lptr0;
			inp->rawbuf_bytes = n;
			uls_regulate_rawbuf(inp);
			uls_resize_rawbuf(inp, txtlen + 1);
			goto again_1;
		}

		insert_txt_record_into_stream(tok_id, xctx->toknum_TMPL, txtptr, txtlen,
			ctx, ss_dst);

		++n_recs;
	}

	inp->rawbuf_ptr = lptr;
	inp->rawbuf_bytes = n;

	parms->n = n_recs;
	return 1;
}

ULS_DECL_STATIC int
__check_rec_boundary_host_order(uls_xcontext_ptr_t xctx, char* buf, int n)
{
	uls_context_ptr_t ctx = xctx->context;
	csz_str_ptr_t  ss_dst = uls_ptr(ctx->zbuf1);

	uls_rd_packet_ptr_t pkt;
	uls_tmplvar_ptr_t tmpl;
	int    i, tok_id, txtlen, reclen;
	const char *txtptr;

	if (n < ULS_BIN_RECHDR_SZ) {
		return 0;
	}

	tok_id = ((uls_int32 *) buf)[0];
	txtlen = ((uls_int32 *) buf)[1];
	txtptr = buf + ULS_BIN_RECHDR_SZ;
	reclen = ULS_BIN_REC_SZ(txtlen);

	if (reclen > n) {
		return -reclen;
	}

	if (tok_id == xctx->toknum_TMPL) {
		if ((tmpl = uls_search_tmpls(uls_ptr(ctx->tmpls_pool->tmplvars), txtptr)) == nilptr || tmpl->n_pkt_ary < 0) {
			add_bin_packet_to_zbuf(tok_id, txtlen, txtptr, ss_dst);
		} else {
			for (i=0; i<tmpl->n_pkt_ary; i++) {
				pkt = uls_array_get_slot_type10(uls_ptr(ctx->tmpls_pool->pkt_ary), tmpl->i0_pkt_ary + i);
				add_bin_packet_to_zbuf(pkt->tok_id, pkt->len_tokstr, pkt->tokstr, ss_dst);
			}
		}
	} else {
		add_bin_packet_to_zbuf(tok_id, txtlen, txtptr, ss_dst);
	}

	// reclen > ULS_BIN_RECHDR_SZ > 0
	return reclen;
}

ULS_DECL_STATIC int
__check_rec_boundary_reverse_order(uls_xcontext_ptr_t xctx, char* buf, int n)
{
	uls_context_ptr_t ctx = xctx->context;
	csz_str_ptr_t  ss_dst = uls_ptr(ctx->zbuf1);

	uls_rd_packet_ptr_t pkt;
	uls_tmplvar_ptr_t tmpl;
	int    i, tok_id, txtlen, reclen;
	const char *txtptr;

	if (n < ULS_BIN_RECHDR_SZ) {
		return 0;
	}

	tok_id = ((uls_int32 *) buf)[0];
	uls_reverse_bytes((char *) &tok_id, sizeof(uls_int32));

	txtlen = ((uls_int32 *) buf)[1];
	uls_reverse_bytes((char *) &txtlen, sizeof(uls_int32));

	if (txtlen < 0)
		err_panic("%s: corrupt stream!", __func__);

	txtptr = buf + ULS_BIN_RECHDR_SZ;
	reclen = ULS_BIN_REC_SZ(txtlen);

	if (reclen > n) {
		return -reclen;
	}

	if (tok_id == xctx->toknum_TMPL) {
		if ((tmpl = uls_search_tmpls(uls_ptr(ctx->tmpls_pool->tmplvars), txtptr)) == nilptr || tmpl->n_pkt_ary < 0) {
			add_bin_packet_to_zbuf(tok_id, txtlen, txtptr, ss_dst);
		} else {
			for (i=0; i<tmpl->n_pkt_ary; i++) {
				pkt = uls_array_get_slot_type10(uls_ptr(ctx->tmpls_pool->pkt_ary), tmpl->i0_pkt_ary + i);
				add_bin_packet_to_zbuf(pkt->tok_id, pkt->len_tokstr, pkt->tokstr, ss_dst);
			}
		}
	} else {
		add_bin_packet_to_zbuf(tok_id, txtlen, txtptr, ss_dst);
	}

	// reclen > ULS_BIN_RECHDR_SZ > 0
	return reclen;
}

ULS_DECL_STATIC int
__check_rec_boundary_bin(uls_xcontext_ptr_t xctx, uls_xctx_boundary_checker2_t checker)
{
	uls_input_ptr_t inp = xctx->context->input;
	int i0 = 0, reclen, n_recs, n, m2;

	if ((n=inp->rawbuf_bytes) < ULS_BIN_RECHDR_SZ) {
		err_log("%s: incorrect format error!", __func__);
		return -1;
	}

	for (n_recs=0; ; n_recs++) {
		if ((reclen = checker(xctx, inp->rawbuf.buf + i0, n)) > 0) {
			i0 += reclen;
			n -= reclen;

		} else if (reclen < 0 && n_recs == 0) {
			reclen = -reclen;

			m2 = reclen - n;  // need m2 more bytes to complete a record.

			str_modify(uls_ptr(inp->rawbuf), inp->rawbuf_bytes, NULL, m2);
			if (uls_input_readn(uls_ptr(inp->isource), inp->rawbuf.buf, inp->rawbuf_bytes, reclen) < m2) {
				err_log("%s: file error", __func__);
				return -1;
			}

			n = reclen;

		} else {
			break;
		}
	}

	inp->rawbuf_bytes = n; // n may be zero!
	inp->rawbuf_ptr = inp->rawbuf.buf + i0;

	return n_recs;
}

ULS_DECL_STATIC uls_commtype_ptr_t
is_commtype_start(uls_xcontext_ptr_t xctx, const char *ptr, int len)
{
	uls_commtype_ptr_t cmt;
	const char *str;
	int i;

	for (i=0; i<xctx->n2_commtypes; i++) {
		cmt = uls_array_get_slot_type01(xctx->commtypes, i);

		if (len < cmt->len_start_mark) continue;

		str = cmt->start_mark;
		if (ptr[0] == str[0]) {
			if (cmt->len_start_mark == 1 || !uls_strncmp(ptr+1,str+1, cmt->len_start_mark-1)) {
				return cmt;
			}
		}
	}

	return nilptr;
}

const char*
uls_xcontext_quotetype_start_mark(uls_xcontext_ptr_t xctx, int tok_id)
{
	uls_quotetype_ptr_t qmt;

	if ((qmt = uls_find_quotetype_by_tokid(xctx->quotetypes, xctx->quotetypes->n, tok_id)) == nilptr) {
		err_log("No matching quote-string for tok-id(%d) found", tok_id);
		return NULL;
	}

	return qmt->start_mark;
}

uls_quotetype_ptr_t
uls_xcontext_find_quotetype(uls_xcontext_ptr_t xctx, const char *ptr, int len)
{
	uls_decl_parray_slots_init(slots_qmt, quotetype, xctx->quotetypes);
	uls_quotetype_ptr_t qmt;
	const char *str;
	int i;

	for (i=0; i<xctx->quotetypes->n; i++) {
		qmt = slots_qmt[i];

		if (len < qmt->len_start_mark) continue;

		str = qmt->start_mark;
		if (ptr[0] == str[0]) {
			if (qmt->len_start_mark == 1 || !uls_strncmp(ptr+1,str+1, qmt->len_start_mark-1)) {
				return qmt;
			}
		}
	}

	return nilptr;
}

int
check_rec_boundary_null(uls_xcontext_ptr_t xctx, uls_parm_line_ptr_t parm_ln)
{
	uls_lexseg_ptr_t lexseg;

	xctx->context->n_lexsegs = 0;
	lexseg = uls_array_get_slot_type10(uls_ptr(xctx->context->lexsegs), 0);
	uls_reset_lexseg(lexseg, 0, 0, -1, -1, nilptr);

	if (parm_ln != nilptr) {
		parm_ln->lptr = "";
	}
	return 0;
}

void
uls_init_lexseg(uls_lexseg_ptr_t lexseg)
{
}

void
uls_deinit_lexseg(uls_lexseg_ptr_t lexseg)
{
}

void
uls_reset_lexseg(uls_lexseg_ptr_t lexseg,
	int offset1, int len1, int offset2, int len2, uls_tokdef_vx_ptr_t e_vx)
{
	lexseg->offset1 = offset1;
	lexseg->len1 = len1;

	if (offset2 < 0) len2 = -1;
	lexseg->offset2 = offset2;
	lexseg->len_text = len2;

	lexseg->tokdef_vx = e_vx;
	lexseg->n_lfs_raw = 0;
}

void
uls_init_context(uls_context_ptr_t ctx, uls_gettok_t gettok, int tok0)
{
	uls_input_ptr_t   inp;
	uls_lexseg_ptr_t lexseg;
	int i;

	ctx->flags = 0;
	csz_init(uls_ptr(ctx->tag), 128);
	ctx->lineno = 1; ctx->delta_lineno = 0;

	uls_init_bytespool(ctx->cnst_nilstr, ULS_CNST_NILSTR_SIZE, 0);
	ctx->input = inp = uls_create_input();

	csz_init(uls_ptr(ctx->zbuf1), ULS_FDZBUF_INITSIZE);
	csz_add_eos(uls_ptr(ctx->zbuf1));
	csz_init(uls_ptr(ctx->zbuf2), ULS_FDZBUF_INITSIZE);

	ctx->lptr = ctx->line = ctx->cnst_nilstr;
	ctx->line_end = ctx->line;

	uls_init_array_type10(uls_ptr(ctx->lexsegs), lexseg, ULS_LEXSEGS_MAX + 1);
	for (i=0; i < ctx->lexsegs.n_alloc; i++) {
		uls_array_alloc_slot_type10(uls_ptr(ctx->lexsegs), lexseg, i);
	}
	ctx->lexsegs.n = ctx->lexsegs.n_alloc;

	ctx->i_lexsegs = ctx->n_lexsegs = 0;
	lexseg = uls_array_get_slot_type10(uls_ptr(ctx->lexsegs), 0);
	uls_reset_lexseg(lexseg, 0, 0, -1, -1, nilptr);

	ctx->gettok = gettok;
	ctx->fill_proc = uls_ref_callback(xcontext_raw_filler);
	ctx->flags |= ULS_CTX_FL_FILL_RAW;
	ctx->record_boundary_checker = uls_ref_callback(check_rec_boundary_null);

	ctx->tmpls_pool = nilptr;

	ctx->tok = tok0;
	ctx->s_val = ctx->tokbuf.buf;
	ctx->s_val_len = ctx->s_val_uchars = 0;

	str_init(uls_ptr(ctx->tokbuf), (ULS_LEXSTR_MAXSIZ+1)<<1);
	ctx->n_digits = ctx->expo = 0;

	ctx->anonymous_uchar_vx = uls_create_tokdef_vx(0, "", nilptr); // 0: nonsense
	ctx->anonymous_uchar_vx->flags |= ULS_VX_ANONYMOUS;
	ctx->user_data = nilptr;
	ctx->prev = nilptr;
}

void
uls_deinit_context(uls_context_ptr_t ctx)
{
	uls_destroy_input(ctx->input);
	ctx->input = nilptr;

	csz_deinit(uls_ptr(ctx->zbuf1));
	csz_deinit(uls_ptr(ctx->zbuf2));

	ctx->fill_proc = uls_ref_callback(xcontext_raw_filler);
	ctx->flags |= ULS_CTX_FL_FILL_RAW;
	ctx->record_boundary_checker = uls_ref_callback(check_rec_boundary_null);

	csz_deinit(uls_ptr(ctx->tag));
	str_free(uls_ptr(ctx->tokbuf));
	ctx->gettok = nilptr;
	ctx->prev = nilptr;

	uls_destroy_tokdef_vx(ctx->anonymous_uchar_vx);
	ctx->anonymous_uchar_vx = nilptr;

	ctx->flags = 0;
	ctx->lineno = 1; ctx->delta_lineno = 0;
	ctx->line = ctx->lptr = ctx->line_end = ctx->cnst_nilstr;

	uls_deinit_array_type10(uls_ptr(ctx->lexsegs), lexseg);
	ctx->i_lexsegs =  ctx->n_lexsegs =  0;

	uls_destroy_tmpl_pool(ctx->tmpls_pool);
	ctx->tmpls_pool = nilptr;

	uls_deinit_bytespool(ctx->cnst_nilstr);
}

void
uls_xcontext_init(uls_xcontext_ptr_t xctx, uls_gettok_t gettok)
{
	uls_initial_zerofy_object(xctx);

	xctx->toknum_EOI = 0;
	xctx->toknum_EOF = -1;
	xctx->toknum_ID  = -2;
	xctx->toknum_NUMBER = -3;
	xctx->toknum_LINENUM = -4;
	xctx->toknum_TMPL = -5;
	xctx->toknum_LINK = -6;
	xctx->toknum_NONE = -7;
	xctx->toknum_ERR = -8;

	xctx->len_surplus = ULS_UTF8_CH_MAXLEN;

	xctx->context = uls_alloc_object(uls_context_t); // initial-context
	uls_init_context(xctx->context, gettok, xctx->toknum_NONE);
	uls_input_set_fl(xctx->context->input, ULS_INP_FL_READONLY);
}

void
uls_xcontext_reset(uls_xcontext_ptr_t xctx)
{
	uls_context_set_line(xctx->context, NULL, -1);
	uls_context_set_tag(xctx->context, "", 1);
}

void
uls_xcontext_deinit(uls_xcontext_ptr_t xctx)
{
	uls_xcontext_reset(xctx);

	xctx->commtypes = nilptr;
	xctx->n2_commtypes = 0;

	xctx->prepended_input = uls_mfree(xctx->prepended_input);
	xctx->len_prepended_input = xctx->lfs_prepended_input = 0;
	xctx->quotetypes = nilptr;

	uls_deinit_context(xctx->context);
	uls_dealloc_object(xctx->context);
	xctx->context = nilptr;
}

int
check_rec_boundary_host_order(uls_xcontext_ptr_t xctx, uls_parm_line_ptr_t parm_ln)
{
	uls_input_ptr_t inp = xctx->context->input;
	uls_callback_type(xctx_boundary_checker2) checker2;
	int rc;

	checker2 = uls_ref_callback(__check_rec_boundary_host_order);
	rc = __check_rec_boundary_bin(xctx, checker2);
	parm_ln->lptr = inp->rawbuf.buf;

	return rc;
}

int
check_rec_boundary_reverse_order(uls_xcontext_ptr_t xctx, uls_parm_line_ptr_t parm_ln)
{
	uls_input_ptr_t inp = xctx->context->input;
	uls_callback_type(xctx_boundary_checker2) checker2;
	int rc;

	checker2 = uls_ref_callback(__check_rec_boundary_reverse_order);
	rc = __check_rec_boundary_bin(xctx, checker2);
	parm_ln->lptr = inp->rawbuf.buf;

	return rc;
}

int
xcontext_raw_filler(uls_xcontext_ptr_t xctx)
{
	uls_context_ptr_t ctx = xctx->context;
	uls_input_ptr_t inp = xctx->context->input;
	const char    *ch_ctx = xctx->ch_context;

	csz_str_ptr_t ss_dst1 = uls_ptr(ctx->zbuf1);
	csz_str_ptr_t ss_dst2 = uls_ptr(ctx->zbuf2);
	uls_lexseg_ptr_t  lexseg;
	const char  *lptr1, *lptr, *lptr_end;
	int   n_segs=0, k2, offset1, rc, len_surplus;
	char  ch, ch_grp;

	uls_commtype_ptr_t cmt;
	uls_quotetype_ptr_t qmt;
	int i, n_lfs;
	uls_outparam_t parms1;

	len_surplus = xctx->len_surplus;
	offset1 = 0;

	lptr1 = lptr = inp->rawbuf_ptr;
	lptr_end = lptr + inp->rawbuf_bytes;

	for ( ; ; ) {
		if (lptr_end < lptr + len_surplus) {
			if ((rc = (int) (lptr-lptr1)) > 0) {
				csz_append(ss_dst1, lptr1, rc);
			}

			inp->rawbuf_ptr = lptr;
			inp->rawbuf_bytes = (int) (lptr_end - lptr);

			if (inp->refill(inp, len_surplus) < 0) {
//				err_log("%s: I/O error", __func__);
				uls_input_reset_cursor(inp);
				return -1;
			}

			lptr1 = lptr = inp->rawbuf_ptr;
			lptr_end = lptr + inp->rawbuf_bytes;

			if (inp->rawbuf_bytes == 0) {
				ctx->flags |= ULS_CTX_FL_EOF;
				break;
			}
		}

		if ((ch=*lptr) < ULS_SYNTAX_TABLE_SIZE) {
			ch_grp = ch_ctx[ch];
		} else {
			ch_grp = 0xFF & ~(ULS_CH_COMM | ULS_CH_QUOTE);
		}

		if (ch_grp == 0) {
			if ((rc = (int) (lptr-lptr1)) > 0) csz_append(ss_dst1, lptr1, rc);
			if (csz_length(ss_dst1) >= ULS_FDBUF_SIZE) break;

			inp->rawbuf_ptr = lptr;
			inp->rawbuf_bytes = (int) (lptr_end - lptr);

			if ((rc=input_space_proc(ch_ctx, inp, ss_dst1, len_surplus, uls_ptr(parms1))) < 0) {
				err_log("%s: I/O error", __func__);
				return -1;
			}

			n_lfs = parms1.n;

			lptr1 = lptr = inp->rawbuf_ptr;
			lptr_end = lptr + inp->rawbuf_bytes;
			inp->line_num += n_lfs;

			if (rc == 0) break;

		} else if ((ch_grp & ULS_CH_COMM) &&
			(cmt=is_commtype_start(xctx, lptr, (int) (lptr_end - lptr))) != nilptr) {
			lptr += cmt->len_start_mark;

			inp->rawbuf_ptr = lptr;
			inp->rawbuf_bytes = (int) (lptr_end - lptr);

			rc = input_skip_comment(cmt, inp, uls_ptr(parms1));
			n_lfs = parms1.n;

			lptr1 = lptr = inp->rawbuf_ptr;
			lptr_end = lptr + inp->rawbuf_bytes;

			if (rc <= 0) {
				err_log("[%s:%d] Unterminated comment at EOF!", uls_context_get_tag(ctx), inp->line_num);
				return -1;
			}

			n_lfs += cmt->n_lfs;
			for (i=0; i < n_lfs; i++) csz_putc(ss_dst1, '\n');
			inp->line_num += n_lfs;

		} else if ((ch_grp & ULS_CH_QUOTE) &&
			(qmt=uls_xcontext_find_quotetype(xctx, lptr, (int) (lptr_end - lptr))) != nilptr) {
			if ((rc = (int) (lptr-lptr1)) > 0) csz_append(ss_dst1, lptr1, rc);
			if (n_segs + 1 >= ctx->lexsegs.n) {
				break;
			}

			lptr += qmt->len_start_mark;
			lexseg = uls_array_get_slot_type10(uls_ptr(ctx->lexsegs), n_segs);
			lexseg->tokdef_vx = qmt->tokdef_vx;

			lexseg->offset1 = offset1;
			lexseg->len1 = csz_length(ss_dst1) - offset1;
			// forcing '\0' before putting literal-string ...
			csz_add_eos(ss_dst1);
			offset1 = csz_length(ss_dst1);

			lexseg->offset2 = k2 = csz_length(ss_dst2);

			inp->rawbuf_ptr = lptr;
			inp->rawbuf_bytes = (int) (lptr_end - lptr);
			rc = input_quote_proc(inp, qmt, ss_dst2, uls_ptr(parms1));
			n_lfs = parms1.n;

			if (rc == ULS_LITPROC_ENDOFQUOTE) {
				lexseg->n_lfs_raw = n_lfs;

				lexseg->len_text = csz_length(ss_dst2) - lexseg->offset2;
				// put '\0' at the end of literal string
				csz_add_eos(ss_dst2);
				++n_segs;

			} else if (rc == ULS_LITPROC_DISMISSQUOTE) {
				lexseg->n_lfs_raw = n_lfs;
				csz_truncate(ss_dst2, k2);

			} else {
				err_log("[%s:%d] Unterminated literal string at EOF!", uls_context_get_tag(ctx), inp->line_num);
				return -1;
			}

			lptr1 = lptr = inp->rawbuf_ptr;
			lptr_end = lptr + inp->rawbuf_bytes;
			inp->line_num += n_lfs;

		} else {
			if (ch == '\n') ++inp->line_num;
			++lptr;
		}
	}

	inp->rawbuf_ptr = lptr;
	inp->rawbuf_bytes = (int) (lptr_end - lptr);

	lexseg = uls_array_get_slot_type10(uls_ptr(ctx->lexsegs), n_segs);
	uls_reset_lexseg(lexseg, offset1, csz_length(ss_dst1) - offset1, -1, -1, nilptr);

	csz_add_eos(ss_dst1);
	ctx->n_lexsegs = n_segs;
	lexseg = uls_array_get_slot_type10(uls_ptr(ctx->lexsegs), 0);

	return lexseg->len1;
}

int
xcontext_binfd_filler(uls_xcontext_ptr_t xctx)
{
	uls_context_ptr_t ctx = xctx->context;
	int rc;
	uls_parm_line_t parm_ln1;

	do {
		if ((rc = __xcontext_binfd_filler(xctx)) <= 0) {
			return rc;
		}

		rc = xctx->context->record_boundary_checker(xctx, uls_ptr(parm_ln1));
		if (rc < 0) {
			return -1;
		}
		// rc : # of recs
	} while (rc == 0);

	return csz_length(uls_ptr(ctx->zbuf1));
}

int
xcontext_txtfd_filler(uls_xcontext_ptr_t xctx)
{
	uls_context_ptr_t ctx = xctx->context;
	uls_outparam_t parms;
	int rc, n_recs;

	do {
		if ((rc = __xcontext_txtfd_filler(xctx, uls_ptr(parms))) < 0) {
			return -1;
		} else if (rc == 0) {
			ctx->flags |= ULS_CTX_FL_EOF;
			break;
		}

		n_recs = parms.n;
	} while (n_recs == 0);

	return csz_length(uls_ptr(ctx->zbuf1));
}

void
uls_context_set_tag(uls_context_ptr_t ctx, const char* tagstr, int lno)
{
	if (tagstr != NULL) {
		csz_reset(uls_ptr(ctx->tag));
		csz_append(uls_ptr(ctx->tag), tagstr, -1);
	}

	if (lno >= 0) {
		ctx->lineno = lno;
	}
}

int
uls_xcontext_delete_litstr_analyzer(uls_xcontext_ptr_t xctx, const char* prefix)
{
	uls_decl_parray_slots_init(slots_qmt, quotetype, xctx->quotetypes);
	uls_quotetype_ptr_t qmt;
	int i, j;

	for (i=0; ; i++) {
		if (i >= xctx->quotetypes->n) {
			return -1;
		}

		qmt = slots_qmt[i];
		if (uls_streql(qmt->start_mark, prefix)) {
			break;
		}
	}

	if (i < xctx->quotetypes->n - 1) {
		qmt = slots_qmt[i];
		uls_destroy_quotetype(qmt);

		for (j=i; j < xctx->quotetypes->n - 1; j++) {
			qmt = slots_qmt[j+1];
			slots_qmt[j] = qmt;
		}
	}

	--xctx->quotetypes->n;
	return 0;
}

int
uls_xcontext_change_litstr_analyzer(uls_xcontext_ptr_t xctx,
	const char* prefix, uls_litstr_analyzer_t lit_analyzer, uls_voidptr_t dat)
{
	uls_decl_parray_slots_init(slots_qmt, quotetype, xctx->quotetypes);
	uls_quotetype_ptr_t qmt;
	int i, stat = -1;

	if (lit_analyzer == nilptr) {
		lit_analyzer = uls_ref_callback(dfl_lit_analyzer_escape0);
	}

	for (i=0; i<xctx->quotetypes->n; i++) {
		qmt = slots_qmt[i];

		if (uls_streql(qmt->start_mark, prefix)) {
			qmt->litstr_analyzer = lit_analyzer;
			qmt->litstr_context = dat;
			stat = 0;
			break;
		}
	}

	return stat;
}

void
uls_context_inc_lineno(uls_context_ptr_t ctx, int delta)
{
	int lno;

	if ((lno = ctx->lineno + delta) >= 0) {
		uls_context_set_tag(ctx, NULL, lno);
	}
}

void
uls_context_set_line(uls_context_ptr_t ctx, const char* line, int len)
{
	if (line == NULL) {
		line = "";
		len = 0;
	} else if (len < 0) {
		len = uls_strlen(line);
	}

	ctx->line = ctx->lptr = line;
	if ((ctx->line_end=line+len) < line)
		err_panic("%s: invalid param(len=%d)!", __func__, len);
}